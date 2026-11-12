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

It trains on a GPU if there is one and on the CPU otherwise. Gradient checkpointing is on by default and the run below peaked at about 9 GB of RAM. `--no-save-memory` is slightly faster, but it reached 14 GB before the longest modules came up. `--max-minutes 100` stops cleanly with a checkpoint so the run can go on later with `--resume`.

**Fix rate** is the share of patches that compile with Icarus Verilog and that Yosys proves equivalent to the original module. **Baseline**: return the buggy input unchanged.

Fix rate can only be measured on modules the tools can judge: the original has to compile on its own and be synthesizable with at least one output. About a third of the dataset qualifies. The rest include other files, use modules defined elsewhere, or are testbenches. For those, `evaluate.py` reports only `exact`: the patch has the same text as the original.

### First run (Nov 12)

One epoch on 1,657 train examples (`--epochs 1 --dropout 0`, LoRA rank 16, CPU, 199 minutes), scored on the test split: 227 buggy modules, 90 of them scorable.

| bug type | scorable | fix rate | baseline | exact (all modules) |
|---|---|---|---|---|
| blocking_in_seq | 18 | 88.9% | 100.0% | 82.6% of 46 |
| inverted_condition | 11 | 72.7% | 0.0% | 81.6% of 38 |
| missing_default | 15 | 40.0% | 60.0% | 4.3% of 23 |
| reset_polarity | 24 | 83.3% | 4.2% | 71.4% of 35 |
| width_off_by_one | 6 | 33.3% | 16.7% | 60.0% of 45 |
| wrong_operator | 16 | 68.8% | 12.5% | 55.0% of 40 |
| **all** | **90** | **70.0%** | **34.4%** | **63.4% of 227** |

- The baseline is high for `blocking_in_seq` and `missing_default` because Yosys compares synthesized hardware: `=` and `<=` usually synthesize to the same circuit, and so does a full `case` without its `default`. For these two, `exact` says more than fix rate.
- 96.9% of the answers could be applied to the module. Of the other seven, six name lines that are not in it and one ran past the length limit.
- When a module has several identical `always @(posedge clk or negedge rst)` lines and one is flipped, the model sometimes flips another one instead of the odd one out.
- The per-type numbers are on 6 to 24 modules each, so they move a lot from one module.

## Rules

- Never commit API keys or passwords. Put them in `.env` (ignored by git).
- Never commit model weights. They go in `checkpoints/` (ignored).
- Both models are scored on the **same test split**: `split == "test"` from `splits.py`.
