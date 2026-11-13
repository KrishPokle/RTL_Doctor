"""
The bug detector: CodeT5+ 220M encoder with a classification head.

Input:  one Verilog module.
Output: clean/buggy and the bug type (one of LABELS), with a confidence.

A module longer than the encoder's window is cut into overlapping windows of
tokens. Each window is classified on its own and the module takes the answer
of the window that is most sure there is a bug, so a bug anywhere in a long
module can still be found.

    from detect.model import load_detector, detect_bugs
    model, tok = load_detector()                  # checkpoints/detect
    detect_bugs(model, tok, code)
    # {'label': 'buggy', 'bug_type': 'reset_polarity', 'confidence': 0.97,
    #  'probs': {...}, 'window_lines': (12, 58)}
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "Salesforce/codet5p-220m"
CHECKPOINT = ROOT / "checkpoints" / "detect"

LABELS = ["none", "blocking_in_seq", "missing_default", "width_off_by_one",
          "inverted_condition", "reset_polarity", "wrong_operator"]
LABEL_ID = {name: i for i, name in enumerate(LABELS)}

WINDOW = 512      # tokens per window (CodeT5+ was pre-trained on 512)
STRIDE = 384      # tokens between window starts: windows overlap by 128


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BugDetector(nn.Module):
    """T5 encoder -> mean over tokens -> dropout -> linear layer over LABELS."""

    def __init__(self, encoder, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.config.d_model, len(LABELS))

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(self.dropout(pooled))


def load_tokenizer():
    """
    CodeT5+'s RoBERTa BPE tokenizer, built from its vocab and merges files.
    Its tokenizer_config.json stores special tokens in an old format that
    transformers 5 can no longer read, so from_pretrained() fails on it.
    """
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    bpe = Tokenizer(models.BPE.from_file(hf_hub_download(BASE_MODEL, "vocab.json"),
                                         hf_hub_download(BASE_MODEL, "merges.txt"), unk_token="<unk>"))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.post_processor = processors.RobertaProcessing(("</s>", bpe.token_to_id("</s>")), ("<s>", bpe.token_to_id("<s>")),
                                                      add_prefix_space=False, trim_offsets=False)
    return PreTrainedTokenizerFast(tokenizer_object=bpe, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                   sep_token="</s>", cls_token="<s>", pad_token="<pad>", mask_token="<mask>")


def new_detector(dropout=0.1):
    """CodeT5+ encoder with a fresh (untrained) head."""
    from transformers import T5EncoderModel
    return BugDetector(T5EncoderModel.from_pretrained(BASE_MODEL, dtype=torch.float32), dropout)


def save_detector(model, out, settings=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "detector.pt")
    (out / "run_config.json").write_text(json.dumps({"base_model": BASE_MODEL, "labels": LABELS,
                                                     **(settings or {})}, indent=2, default=str))


def load_detector(path=CHECKPOINT, device=None):
    """(model, tokenizer) from a folder written by detect.train."""
    from transformers import T5Config, T5EncoderModel
    path = Path(path)
    if not (path / "detector.pt").exists():
        raise FileNotFoundError(f"no trained detector in {path}: run `python -m detect.train` first")
    saved = json.loads((path / "run_config.json").read_text())
    if saved.get("labels", LABELS) != LABELS:
        raise ValueError(f"{path} was trained on different labels: {saved['labels']}")
    device = device or pick_device()
    model = BugDetector(T5EncoderModel(T5Config.from_pretrained(BASE_MODEL)))
    model.load_state_dict(torch.load(path / "detector.pt", map_location="cpu"))
    return model.to(device).eval(), load_tokenizer()


# ------------------------------------------------------------------ windows
def windows(tok, code, window=WINDOW, stride=STRIDE):
    """
    Token windows of one module: list of (ids, first_char, last_char).
    Each window holds at most `window` tokens including the special tokens.
    """
    enc = tok(code, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc.input_ids, enc.offset_mapping
    room = window - 2                                   # <s> ... </s>
    starts = list(range(0, max(len(ids) - room, 0) + 1, stride))
    if starts[-1] + room < len(ids):                    # make sure the tail is covered
        starts.append(len(ids) - room)
    out = []
    for s in starts:
        piece = ids[s:s + room]
        first = offsets[s][0] if piece else 0
        last = offsets[s + len(piece) - 1][1] if piece else 0
        out.append(([tok.cls_token_id] + piece + [tok.sep_token_id], first, last))
    return out


def line_span(code, first_char, last_char):
    """1-based (first_line, last_line) covered by a character range."""
    return code.count("\n", 0, first_char) + 1, code.count("\n", 0, last_char) + 1


def window_with_line(tok, code, line, window=WINDOW, stride=STRIDE):
    """The ids of the window that holds `line` nearest its middle (all windows hold it if line < 1)."""
    wins = windows(tok, code, window, stride)
    if line < 1 or len(wins) == 1:
        return wins[0][0]
    best, best_gap = wins[0][0], None
    for ids, a, b in wins:
        lo, hi = line_span(code, a, b)
        if lo <= line <= hi:
            gap = abs(line - (lo + hi) / 2)
            if best_gap is None or gap < best_gap:
                best, best_gap = ids, gap
    return best


def pad_batch(seqs, pad_id, device):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


# ------------------------------------------------------------------ inference
@torch.no_grad()
def predict_many(model, tok, codes, batch_size=16, window=WINDOW, stride=STRIDE):
    """detect_bugs() for many modules, batching windows across modules."""
    model.eval()
    device = next(model.parameters()).device
    owners, seqs, spans = [], [], []
    for m, code in enumerate(codes):
        for ids, a, b in windows(tok, code, window, stride):
            owners.append(m)
            seqs.append(ids)
            spans.append(line_span(code, a, b))
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))    # less padding
    probs = [None] * len(seqs)
    for k in range(0, len(order), batch_size):
        chunk = order[k:k + batch_size]
        ids, mask = pad_batch([seqs[i] for i in chunk], tok.pad_token_id, device)
        for i, p in zip(chunk, F.softmax(model(ids, mask).float(), -1).cpu()):
            probs[i] = p
    results = []
    for m in range(len(codes)):
        mine = [i for i, o in enumerate(owners) if o == m]
        # the window least sure the module is clean speaks for the module
        w = min(mine, key=lambda i: probs[i][0].item())
        p = probs[w]
        k = int(p.argmax())
        results.append({
            "label": "clean" if k == 0 else "buggy",
            "bug_type": LABELS[k],
            "confidence": round(p[k].item(), 4),
            "probs": {name: round(p[j].item(), 4) for j, name in enumerate(LABELS)},
            "window_lines": spans[w],
        })
    return results


def detect_bugs(model, tok, code):
    """Is this Verilog module buggy, and with which bug type? See the module docstring."""
    return predict_many(model, tok, [code])[0]
