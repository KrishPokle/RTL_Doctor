"""
Load deepseek-coder (with or without the LoRA adapter) and generate repairs.

    model, tok = load("checkpoints/repair")
    patched = repair_rtl(model, tok, code, "blocking_in_seq")
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from repair.prompt import apply_answer, build_prompt

BASE_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"


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
def generate_answers(model, tok, codes, bug_types, fmt="edit", batch_size=4, progress=None):
    """Raw model answers for a list of buggy modules (greedy decoding)."""
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
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        for i, row in zip(idx, out):
            answers[i] = tok.decode(row[width:], skip_special_tokens=True)
        if progress:
            progress(min(start + batch_size, len(order)), len(order))
    return answers


def repair_rtl(model, tok, code, bug_type, fmt="edit"):
    """Patched module for one buggy module, or None if the model's answer is unusable."""
    answer = generate_answers(model, tok, [code], [bug_type], fmt)[0]
    return apply_answer(code, answer, fmt)
