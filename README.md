# RTL Doctor

An agent that finds, explains, and repairs bugs in Verilog RTL designs, and draws them as gate-level schematics.

| Who | Part |
|---|---|
| Pranav Senthilkumar | Bug detection: fine-tuned CodeT5+ |
| Krish Pokle | Bug repair: fine-tuned deepseek-coder; tools and portal |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python make_dataset.py           # builds data/rtl_bugs.csv (a few minutes)
```

The dataset is **not** in the repo. `make_dataset.py` rebuilds it, and its fixed random seed gives everyone the same 3,000 rows.

Verilog tools (iverilog, Yosys, Verilator) come from the [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build). On Ubuntu, `sudo apt install iverilog yosys` is enough for the repair model.

## Files

| File | What it is |
|---|---|
| `make_dataset.py` | Pulls Verilog from [shailja/Verilog_GitHub](https://huggingface.co/datasets/shailja/Verilog_GitHub) and injects one bug per module |
| `rtl_dataset.ipynb` | Dataset overview: `df.head()`, bug-type plot, baseline |
| `splits.py` | The train / val / test split that both models use |
| `rtl_tools.py` | Icarus Verilog and Yosys wrappers: does a module compile, do two modules behave the same |
| `repair/prompt.py` | What the repair model reads and writes |
| `repair/train.py` | LoRA fine-tune of deepseek-coder-1.3b-instruct |
| `repair/evaluate.py` | Fix rate on the test split, next to the baseline |
| `repair/model.py` | Loads the model and adapter; `repair_rtl(model, tok, code, bug_type)` |
| `tests/` | `python -m pytest tests` |
| `docs/Project_M1_Proposal.pdf` | The M1 project proposal: models, tools, portal, guardrails, Nov 20 targets |

## Dataset

One row = one Verilog module.

| Column | Meaning |
|---|---|
| `module` | Module name |
| `code` | The Verilog (buggy or clean) |
| `label` | `clean` or `buggy` |
| `bug_type` | `blocking_in_seq`, `missing_default`, `width_off_by_one`, `inverted_condition`, `reset_polarity`, `wrong_operator`, or `none` |
| `bug_line` | Line the bug is on (`-1` if clean) |
| `n_lines` | Length of the module |
| `original_code` | The correct version (the repair model's target) |

## Train / val / test split

`splits.py` puts each module in `train` (80%), `val` (10%) or `test` (10%) from a hash of its original code, so the split is identical on every machine and for both models.

```python
import pandas as pd
from splits import add_split

df = add_split(pd.read_csv("data/rtl_bugs.csv"))
test = df[df.split == "test"]
```

## Repair model

deepseek-coder-1.3b-instruct with a LoRA adapter. It reads the buggy module and the bug type and writes one SEARCH/REPLACE block (the buggy lines, then the fixed lines), which `repair/prompt.py` applies to get the patched module. `--format module` makes it write the whole corrected module instead; that is about 15 times more text to generate per fix.

```bash
python -m repair.train                    # 3 epochs on the buggy rows of the train split
python -m repair.train --resume           # continue a run that was stopped
python -m repair.evaluate                 # fix rate on the test split, next to the baseline
python -m repair.evaluate --no-model      # baseline only
```

It trains on a GPU if there is one and on the CPU otherwise. On a CPU, add `--max-minutes 110` to stop cleanly with a checkpoint, then `--resume`.

**Fix rate** is the share of patches that compile with Icarus Verilog and that Yosys proves equivalent to the original module. **Baseline**: return the buggy input unchanged.

Fix rate can only be measured on modules the tools can judge: the original has to compile on its own and be synthesizable with at least one output. About a third of the dataset qualifies. The rest include other files, use modules defined elsewhere, or are testbenches. For those, `evaluate.py` reports only `exact`: the patch has the same text as the original.

## Rules

- Never commit API keys or passwords. Put them in `.env` (ignored by git).
- Never commit model weights. They go in `checkpoints/` (ignored).
- Both models are scored on the **same test split**: `split == "test"` from `splits.py`.
