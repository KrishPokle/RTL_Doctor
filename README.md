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

Verilog tools (iverilog, Yosys, Verilator) come from the [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build). On Ubuntu, `sudo apt install iverilog yosys verilator` covers the repair model and the linter. The schematic also needs [netlistsvg](https://github.com/nturley/netlistsvg): `npm install -g netlistsvg` (it runs on Node.js).

## Files

| File | What it is |
|---|---|
| `make_dataset.py` | Pulls Verilog from [shailja/Verilog_GitHub](https://huggingface.co/datasets/shailja/Verilog_GitHub) and injects one bug per module |
| `rtl_dataset.ipynb` | Dataset overview: `df.head()`, bug-type plot, baseline |
| `splits.py` | The train / val / test split that both models use |
| `rtl_tools.py` | Icarus Verilog, Verilator, Yosys and netlistsvg wrappers: does a module compile, lint it, do two modules behave the same, draw it as a schematic |
| `repair/prompt.py` | What the repair model reads and writes |
| `repair/train.py` | LoRA fine-tune of deepseek-coder-1.3b-instruct |
| `repair/evaluate.py` | Fix rate on the test split, next to the baseline |
| `repair/model.py` | Loads the model and adapter; `repair_rtl(model, tok, code, bug_type)` |
| `portal/app.py` | The web portal: `python -m portal.app` |
| `portal/agent.py` | One chat turn: system prompt, the tools the model may call, the loop that runs them |
| `portal/llm.py` | The chat model: any OpenAI-compatible API, or a built-in stand-in |
| `portal/guardrails.py` | Input guardrails: which files are accepted, the prompt-injection filter, the topic check |
| `portal/static/` | The page: chat on the left, dashboard on the right |
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

A fix has to apply to the module, change it, and not break a module that compiled. `repair_rtl` tries the model's first answer; if that fails, it tries the answers of a 4-beam search in order, and returns `None` when none is usable.

**Fix rate** is the share of modules that, after the repair, compile with Icarus Verilog and that Yosys proves equivalent to the original. A module without a usable fix stays as it was. **Baseline**: return the buggy input unchanged.

Fix rate can only be measured on modules the tools can judge: the original has to compile on its own and be synthesizable with at least one output. About a third of the dataset qualifies. The rest include other files, use modules defined elsewhere, or are testbenches. For those, `evaluate.py` reports only `exact`: the patch has the same text as the original.

### First run (Nov 12)

One epoch on 1,657 train examples (`--epochs 1 --dropout 0`, LoRA rank 16, CPU, 199 minutes), scored on the test split: 227 buggy modules, 90 of them scorable.

| bug type | scorable | fix rate | baseline | exact (all modules) |
|---|---|---|---|---|
| blocking_in_seq | 18 | 94.4% | 100.0% | 84.8% of 46 |
| inverted_condition | 11 | 72.7% | 0.0% | 84.2% of 38 |
| missing_default | 15 | 60.0% | 60.0% | 4.3% of 23 |
| reset_polarity | 24 | 83.3% | 4.2% | 71.4% of 35 |
| width_off_by_one | 6 | 33.3% | 16.7% | 60.0% of 45 |
| wrong_operator | 16 | 68.8% | 12.5% | 57.5% of 40 |
| **all** | **90** | **74.4%** | **34.4%** | **64.8% of 227** |

- The baseline is high for `blocking_in_seq` and `missing_default` because Yosys compares synthesized hardware: `=` and `<=` usually synthesize to the same circuit, and so does a full `case` without its `default`. For these two, `exact` says more than fix rate.
- The model's first answer was usable for 213 of the 227 modules. The second try found a usable fix for 8 more, and 6 stay unchanged. Before the compile check and the second try were added, the same adapter scored 70.0%.
- When a module has several identical `always @(posedge clk or negedge rst)` lines and one is flipped, the model sometimes flips another one instead of the odd one out.
- The per-type numbers are on 6 to 24 modules each, so they move a lot from one module.

## Portal

```bash
python -m portal.app        # then open http://127.0.0.1:8000
```

Choose a `.v` or `.sv` file (up to 100 kB) and ask about it. The chat runs the tools and the dashboard shows what they return:

| Question | Tool | Dashboard tab |
|---|---|---|
| Can you detect any bugs in my design? | `lint_rtl` (the CodeT5+ detector will join it) | Findings: flagged source lines, findings by check |
| Can you visualize it as a gate-level schematic? | `render_schematic` | Schematic |
| Can you fix the bug? | `repair_rtl` | Repair: the change, with buttons to use or download the repaired file |

**The chat model.** Copy `.env.example` to `.env` and set `LLM_BASE_URL` (an OpenAI-compatible API: the address that `/chat/completions` is added to), `LLM_API_KEY` and `LLM_MODEL`. `python -m portal.llm` lists the model names the API offers. The model must support tool calling. With no `LLM_BASE_URL` a stand-in answers: it picks the tools from keywords in the question and reports what they returned, so the portal runs end to end without a key, but it cannot explain anything.

**The repair model** is loaded on the first repair. It needs PyTorch and the adapter in `checkpoints/repair/`; without them the chat says so and the rest still works. On a CPU the first repair takes a few minutes.

The tools always run on the uploaded file: the model cannot pass them other code. The page keeps the conversation and the file; the server stores nothing.

### Guardrails

| Guardrail in the proposal | Where | What it does |
|---|---|---|
| Accept only `.v`/`.sv` files, size limit | `guardrails.check_file` | Refuses other types, files over 100 kB, non-text, and a file that names an absolute path (`` `include "/etc/passwd"``), since the tools would open it. |
| Refuse non-RTL questions | `guardrails.on_topic`, then the system prompt | A question passes if it uses hardware words, talks about the file, a bug or a fix, or is a short follow-up. Anything else is refused before the model sees it. |
| Prompt-injection filter | `guardrails.filter_injection` | Comments and strings that read like instructions to the assistant are replaced in the copy the chat model reads, and the page says which lines. The tools still run on the file as uploaded. |
| Sandbox and timeout on tool runs | `rtl_tools.py` | Every run gets a timeout and its own empty folder, eight folders deep, so a relative path (`../../x.v`) cannot leave it. This is not operating-system isolation: run the portal in a container if strangers can reach it. |
| Never show a fix that does not compile | `repair.prompt.usable_patch` | See Repair model. |
| User confirms before an email is sent | not built | There is no email step yet. |

The three input rules are fixed patterns, so rephrasing can get past them; the model's own instructions are the second layer. On the 3,000 dataset modules, `check_file` refuses 11 (all name an absolute path) and the injection filter flags none.

## Lint

`lint_rtl(code)` in `rtl_tools.py` runs Verilator's linter and returns a list of findings, each with `line`, `col`, `severity` (`warning` or `error`), `code` (Verilator's name for the check, such as `BLKSEQ`) and `message`. A clean module gives an empty list.

```bash
python rtl_tools.py design.v --lint
```

What it sees of the injected bugs, on 300 buggy modules from the dataset (counting only modules whose original lints without an error):

| bug type | bugs that add a lint finding | Verilator check |
|---|---|---|
| blocking_in_seq | 21 of 29 | `BLKSEQ`, `BLKANDNBLK` |
| missing_default | 17 of 26 | `CASEINCOMPLETE` |
| wrong_operator | 9 of 25 | `SELRANGE`, `WIDTHEXPAND` |
| width_off_by_one | 6 of 17 | `WIDTHTRUNC`, `WIDTHEXPAND` |
| inverted_condition | 1 of 29 | |
| reset_polarity | 0 of 36 | |

## Schematic

`render_schematic(code, level)` in `rtl_tools.py` has Yosys synthesize the module and netlistsvg draw it. It returns the SVG text, or `None` with the reason.

```bash
python rtl_tools.py design.v               # writes design.svg: single gates and flip-flops
python rtl_tools.py design.v --level rtl   # adders, comparators, multiplexers and registers as blocks
```

- A drawing is refused above 500 cells. Level `rtl` keeps buses and arithmetic as single blocks, so it still works for most designs that are too big to draw gate by gate.
- A module defined in another file is drawn as an empty box.
- Yosys gets 10 seconds and netlistsvg 30. Names from the Verilog are escaped before they go into the SVG.

## Rules

- Never commit API keys or passwords. Put them in `.env` (ignored by git).
- Never commit model weights. They go in `checkpoints/` (ignored).
- Both models are scored on the **same test split**: `split == "test"` from `splits.py`.
