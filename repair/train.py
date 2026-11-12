"""
LoRA fine-tune deepseek-coder-1.3b-instruct to repair Verilog bugs.

Trains on the buggy rows of the train split: (buggy module + bug type) -> fix.

    python -m repair.train                        # full run (3 epochs)
    python -m repair.train --epochs 1 --limit 400 # quick first run
    python -m repair.train --resume               # continue a run that was stopped

Output goes to checkpoints/repair/ (ignored by git):
    adapter_model.safetensors, adapter_config.json   the LoRA adapter
    run_config.json                                  the settings of this run
    train_log.csv                                    loss curve
    trainer_state.pt                                 optimizer state, for --resume
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model

from repair.model import load_base, pick_device, prompt_ids
from repair.prompt import FORMATS, build_answer
from splits import add_split

ROOT = Path(__file__).resolve().parent.parent
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]     # the attention layers
RUN_SETTINGS = ["format", "epochs", "limit", "lr", "rank", "dropout", "save_memory", "accum", "max_tokens",
                "val_examples", "seed"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data", default=ROOT / "data" / "rtl_bugs.csv", type=Path)
    p.add_argument("--out", default=ROOT / "checkpoints" / "repair", type=Path)
    p.add_argument("--format", default="edit", choices=FORMATS, help="what the model writes (see repair/prompt.py)")
    p.add_argument("--epochs", default=3, type=int)
    p.add_argument("--limit", default=None, type=int, help="train on only this many examples")
    p.add_argument("--lr", default=2e-4, type=float)
    p.add_argument("--rank", default=16, type=int, help="LoRA rank")
    p.add_argument("--dropout", default=0.05, type=float, help="LoRA dropout (0 uses less memory)")
    p.add_argument("--save-memory", action=argparse.BooleanOptionalAction, default=None,
                   help="gradient checkpointing: far less memory, ~40%% more time (default: on for a GPU)")
    p.add_argument("--accum", default=8, type=int, help="examples per optimizer step")
    p.add_argument("--max-tokens", default=2048, type=int, help="skip examples longer than this")
    p.add_argument("--save-every", default=25, type=int, help="optimizer steps between checkpoints")
    p.add_argument("--val-examples", default=40, type=int, help="val rows used for the loss at each checkpoint")
    p.add_argument("--max-minutes", default=None, type=float, help="stop and save after this long (use --resume to go on)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", default=0, type=int)
    return p.parse_args()


def encode(tok, rows, fmt, max_tokens):
    """(input_ids, labels) per example. The loss is only on the answer, not on the prompt."""
    examples, skipped = [], 0
    for r in rows.itertuples():
        answer = build_answer(r.code, r.original_code, fmt)
        if answer is None:
            skipped += 1
            continue
        prompt = prompt_ids(tok, r.code, r.bug_type, fmt)
        target = tok(answer, add_special_tokens=False).input_ids + [tok.eos_token_id]
        if len(prompt) + len(target) > max_tokens:
            skipped += 1
            continue
        examples.append((prompt + target, [-100] * len(prompt) + target))
    return examples, skipped


def loss_of(model, example):
    """
    Cross-entropy on the answer tokens of one example.
    Same value as model(input_ids, labels).loss, but only the answer positions go
    through the vocabulary layer, which saves memory and time on long prompts.
    """
    ids, labels = example
    n = labels.count(-100)                              # prompt length: the answer starts here
    lm = model.get_base_model()                         # deepseek-coder with the LoRA layers inside
    hidden = lm.base_model(input_ids=torch.tensor([ids], device=model.device)).last_hidden_state[0]
    logits = lm.get_output_embeddings()(hidden[n - 1:-1])       # position i predicts token i + 1
    return F.cross_entropy(logits.float(), torch.tensor(ids[n:], device=model.device))


@torch.no_grad()
def val_loss(model, examples):
    model.eval()
    total = sum(loss_of(model, ex).item() for ex in examples)
    model.train()
    return total / max(len(examples), 1)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    state_file, log_file = args.out / "trainer_state.pt", args.out / "train_log.csv"
    if args.resume:
        if not state_file.exists():
            raise SystemExit(f"--resume: no run to continue in {args.out}")
        # the run's own settings decide the example order and the schedule, so reuse them
        saved = json.loads((args.out / "run_config.json").read_text())
        for key in RUN_SETTINGS:
            setattr(args, key, saved[key])

    # ---- data: buggy rows only, split shared with the detector
    df = add_split(pd.read_csv(args.data))
    df = df[df.label == "buggy"]
    model, tok = load_base()
    if args.save_memory is None:
        args.save_memory = pick_device()[0] == "cuda"      # a 16 GB GPU cannot hold a long module otherwise
    if args.save_memory:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train, skipped = encode(tok, df[df.split == "train"], args.format, args.max_tokens)
    val, _ = encode(tok, df[df.split == "val"].head(args.val_examples), args.format, args.max_tokens)
    random.Random(args.seed).shuffle(train)
    if args.limit:
        train = train[:args.limit]
    tokens = sum(len(ids) for ids, _ in train)
    print(f"device {pick_device()}  |  {len(train)} train examples ({skipped} skipped), "
          f"{len(val)} val, avg {tokens // max(len(train), 1)} tokens")

    # ---- model: frozen base + trainable LoRA adapter
    if args.resume:
        model = PeftModel.from_pretrained(model, args.out, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=args.dropout,
                                                 target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.train()
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    # ---- schedule: the same example order every time, so --resume can pick up where it stopped
    order = []
    for epoch in range(args.epochs):
        epoch_order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(epoch_order)
        order += epoch_order
    total_steps = math.ceil(len(order) / args.accum)
    warmup = max(1, int(0.05 * total_steps))

    def lr_scale(step):             # linear warm-up, then cosine decay to zero
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    done, elapsed_before = 0, 0.0       # examples already trained on, time already spent
    if args.resume:
        state = torch.load(state_file)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        done, elapsed_before = state["done"], state["elapsed"]
        print(f"resuming after {done} examples")
    else:
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(["step", "examples", "epoch", "train_loss", "val_loss", "lr", "minutes"])
        config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        (args.out / "run_config.json").write_text(json.dumps(config, indent=2))

    def save(step, train_loss):
        elapsed = elapsed_before + time.time() - start
        model.save_pretrained(args.out)
        torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "done": done, "elapsed": elapsed}, state_file)
        v = val_loss(model, val) if val else float("nan")
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([step, done, f"{done / len(train):.2f}", f"{train_loss:.4f}",
                                    f"{v:.4f}", f"{sched.get_last_lr()[0]:.2e}", f"{elapsed / 60:.1f}"])
        print(f"  saved  step {step}/{total_steps}  train loss {train_loss:.4f}  val loss {v:.4f}", flush=True)

    # ---- train
    start, recent, since_save, step = time.time(), [], [], done // args.accum
    opt.zero_grad()
    while done < len(order):
        loss = loss_of(model, train[order[done]])
        (loss / args.accum).backward()
        recent.append(loss.item())
        done += 1
        if done % args.accum and done < len(order):
            continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        step += 1
        minutes = (time.time() - start) / 60
        print(f"step {step}/{total_steps}  loss {sum(recent) / len(recent):.4f}  {minutes:.1f} min", flush=True)
        since_save += recent
        recent = []
        out_of_time = args.max_minutes is not None and minutes >= args.max_minutes
        if step % args.save_every == 0 or done == len(order) or out_of_time:
            save(step, sum(since_save) / len(since_save))
            since_save = []
        if out_of_time and done < len(order):
            print(f"stopped after {minutes:.0f} min at {done}/{len(order)} examples; "
                  f"run again with --resume to continue")
            return
    print(f"done: adapter saved to {args.out}")


if __name__ == "__main__":
    main()
