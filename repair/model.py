"""
Load deepseek-coder (with or without the LoRA adapter) and generate repairs.

    model, tok = load("checkpoints/repair")
    patched = repair_rtl(model, tok, code, "blocking_in_seq")     # None: no usable fix

A fix has to apply to the module and must not break its build (Icarus Verilog).
The model's first answer is tried first; if it fails, the answers of a beam
search are tried in order.
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from repair.prompt import build_prompt, usable_patch

BASE_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"
BEAMS = 4           # answers tried when the first one is not usable


def pick_device():
    """(device, dtype): a GPU if there is one, otherwise the CPU."""
    if torch.cuda.is_available():
        return "cuda", (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)
    # 16-bit floats are ~2x faster than 32-bit on CPUs with AMX / AVX512-BF16, and slow on others
    try:
        fast_bf16 = any(f in Path("/proc/cpuinfo").read_text() for f in ("amx_bf16", "avx512_bf16"))
    except OSError:
        fast_bf16 = False
    return "cpu", (torch.bfloat16 if fast_bf16 else torch.float32)


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"           # so a batch of prompts all end where generation starts
    return tok


def load_base():
    device, dtype = pick_device()
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(device)
    return model, load_tokenizer()


def load(adapter=None):
    """The base model, plus the fine-tuned LoRA adapter if a folder is given."""
    model, tok = load_base()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), tok


def adapter_format(adapter):
    """The answer format an adapter was trained with (saved by repair.train)."""
    cfg = Path(adapter, "run_config.json")
    return json.loads(cfg.read_text())["format"] if cfg.exists() else "edit"


def prompt_ids(tok, code, bug_type, fmt):
    """Token ids of the chat prompt, ending where the model's answer begins."""
    text = tok.apply_chat_template([{"role": "user", "content": build_prompt(code, bug_type, fmt)}],
                                   add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False).input_ids


@torch.no_grad()
def generate_answers(model, tok, codes, bug_types, fmt="edit", batch_size=4, beams=1, progress=None):
    """
    Raw model answers for a list of buggy modules: `beams` answers per module, best first.
    One beam is greedy decoding. More beams is a beam search that returns every beam.
    """
    prompts = [prompt_ids(tok, c, b, fmt) for c, b in zip(codes, bug_types)]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))   # similar lengths together
    answers = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = tok.pad({"input_ids": [prompts[i] for i in idx]}, return_tensors="pt").to(model.device)
        width = batch["input_ids"].shape[1]
        # an edit is a few lines; a full module is about as long as its input
        new_tokens = 256 if fmt == "edit" else max(len(tok(codes[i]).input_ids) for i in idx) + 64
        out = model.generate(**batch, max_new_tokens=new_tokens, do_sample=False,
                             num_beams=beams, num_return_sequences=beams,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, width:], skip_special_tokens=True)
        for n, i in enumerate(idx):
            answers[i] = texts[n * beams:(n + 1) * beams]
        if progress:
            progress(min(start + batch_size, len(order)), len(order))
    return answers


def repair_many(model, tok, codes, bug_types, fmt="edit", batch_size=4, beams=BEAMS, progress=None):
    """
    Repair a list of buggy modules. Per module: (patch, answer, attempt).

    attempt 0 is the greedy answer. Where that gives no usable patch (see
    repair.prompt.usable_patch), the answers of a beam search are tried in
    order: attempt 1, 2, ... If none is usable, patch and attempt are None
    and answer is the greedy one.
    """
    results = []
    for code, (answer,) in zip(codes, generate_answers(model, tok, codes, bug_types, fmt, batch_size,
                                                       progress=progress)):
        patch = usable_patch(code, answer, fmt)
        results.append((patch, answer, None if patch is None else 0))
    again = [i for i, (patch, _, _) in enumerate(results) if patch is None]
    if again and beams > 1:
        # one module at a time: a beam search holds `beams` copies of the prompt
        tries = generate_answers(model, tok, [codes[i] for i in again], [bug_types[i] for i in again], fmt,
                                 batch_size=1, beams=beams, progress=progress)
        for i, answers in zip(again, tries):
            for attempt, answer in enumerate(answers, start=1):
                patch = usable_patch(codes[i], answer, fmt)
                if patch is not None:
                    results[i] = (patch, answer, attempt)
                    break
    return results


def repair_rtl(model, tok, code, bug_type, fmt="edit"):
    """Patched module for one buggy module, or None if the model has no usable fix."""
    return repair_many(model, tok, [code], [bug_type], fmt)[0][0]
