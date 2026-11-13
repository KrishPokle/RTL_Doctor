"""
Fine-tune the CodeT5+ 220M encoder to detect Verilog bugs.

Trains on every row of the train split (clean and buggy) to predict the bug
type, with "none" for clean modules. A long module is cut to the token window
that holds its bug line, so the label always matches what the model sees.

    python -m detect.train                        # full run (3 epochs)
    python -m detect.train --epochs 1 --limit 300 # quick first run
    python -m detect.train --max-minutes 90       # stop cleanly; the best checkpoint is kept

Output goes to checkpoints/detect/ (ignored by git):
    detector.pt          weights of the best epoch (highest val macro-F1)
    run_config.json      the settings of this run and its val scores
    train_log.csv        loss and val macro-F1 per epoch
"""
import argparse
import csv
import random
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from detect.model import (LABEL_ID, LABELS, STRIDE, WINDOW, load_tokenizer, new_detector, pad_batch,
                          pick_device, predict_many, save_detector, window_with_line)
from splits import add_split

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data", default=ROOT / "data" / "rtl_bugs.csv", type=Path)
    p.add_argument("--out", default=ROOT / "checkpoints" / "detect", type=Path)
    p.add_argument("--epochs", default=3, type=int)
    p.add_argument("--limit", default=None, type=int, help="train on only this many rows")
    p.add_argument("--lr", default=3e-5, type=float, help="encoder learning rate")
    p.add_argument("--head-lr", default=1e-3, type=float, help="classification head learning rate")
    p.add_argument("--batch-size", default=8, type=int)
    p.add_argument("--window", default=WINDOW, type=int, help="tokens per window")
    p.add_argument("--dropout", default=0.1, type=float)
    p.add_argument("--val-limit", default=None, type=int, help="score only this many val rows per epoch")
    p.add_argument("--max-minutes", default=None, type=float, help="stop after this long (best epoch so far is kept)")
    p.add_argument("--seed", default=0, type=int)
    return p.parse_args()


def encode(tok, rows, window):
    """(ids, label id) per row: the window around the bug line, or the start of a clean module."""
    return [(window_with_line(tok, r.code, int(r.bug_line), window, min(STRIDE, window - 2)), LABEL_ID[r.bug_type])
            for r in rows.itertuples()]


def macro_f1(model, tok, rows, window):
    preds = [p["bug_type"] for p in predict_many(model, tok, list(rows.code), window=window,
                                                 stride=min(STRIDE, window - 2))]
    return f1_score(rows.bug_type, preds, labels=LABELS, average="macro", zero_division=0)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()

    df = add_split(pd.read_csv(args.data))
    train = df[df.split == "train"].sample(frac=1, random_state=args.seed)
    if args.limit:
        train = train.head(args.limit)
    val = df[df.split == "val"]
    if args.val_limit:
        val = val.sample(n=min(args.val_limit, len(val)), random_state=args.seed)

    tok = load_tokenizer()
    examples = encode(tok, train, args.window)
    model = new_detector(args.dropout).to(device)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": args.lr},
                             {"params": model.head.parameters(), "lr": args.head_lr}], weight_decay=0.01)
    steps = args.epochs * ((len(examples) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(steps // 10, 1))
                                              * max(0.0, 1 - s / steps))
    print(f"{len(examples)} train rows, {len(val)} val rows, {steps} steps on {device}")

    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "train_log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "train_loss", "val_macro_f1", "minutes"])
    start, best, stopped = time.time(), -1.0, False
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(examples)
        total, n = 0.0, 0
        for k in range(0, len(examples), args.batch_size):
            batch = examples[k:k + args.batch_size]
            ids, mask = pad_batch([e[0] for e in batch], tok.pad_token_id, device)
            loss = F.cross_entropy(model(ids, mask), torch.tensor([e[1] for e in batch], device=device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, n = total + loss.item() * len(batch), n + len(batch)
            if (k // args.batch_size) % 20 == 0:
                print(f"epoch {epoch}  {n}/{len(examples)}  loss {total / n:.4f}  "
                      f"{(time.time() - start) / 60:.1f} min", flush=True)
            if args.max_minutes and time.time() - start > args.max_minutes * 60:
                stopped = True
                break
        f1 = macro_f1(model, tok, val, args.window)
        minutes = (time.time() - start) / 60
        writer.writerow([epoch, round(total / max(n, 1), 4), round(f1, 4), round(minutes, 1)])
        log.flush()
        print(f"epoch {epoch} done: train loss {total / max(n, 1):.4f}, val macro-F1 {f1:.4f}")
        if f1 > best:
            best = f1
            save_detector(model, args.out, {**vars(args), "best_epoch": epoch, "val_macro_f1": round(f1, 4)})
            print(f"saved to {args.out}")
        if stopped:
            print(f"stopped after {args.max_minutes} minutes")
            break
    log.close()


if __name__ == "__main__":
    main()
