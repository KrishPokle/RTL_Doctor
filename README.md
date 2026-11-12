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

Verilog tools (iverilog, Yosys, Verilator) come from the [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build).

## Files

| File | What it is |
|---|---|
| `make_dataset.py` | Pulls Verilog from [shailja/Verilog_GitHub](https://huggingface.co/datasets/shailja/Verilog_GitHub) and injects one bug per module |
| `rtl_dataset.ipynb` | Dataset overview: `df.head()`, bug-type plot, baseline |
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

## Rules

- Never commit API keys or passwords. Put them in `.env` (ignored by git).
- Never commit model weights. They go in `checkpoints/` (ignored).
- Both models are scored on the **same test split**.
