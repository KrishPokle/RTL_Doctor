# RTL Doctor: development agent guide

Instructions for coding agents (Codex reads this file; Claude Code reads
`CLAUDE.md`, which points here). The agents help develop this repo. They are
not part of the RTL Doctor product.

Project facts come from `README.md` and `docs/Project_M1_Proposal.pdf`. When in
doubt, those documents win over anything an agent assumes.

## Commands

```bash
python make_dataset.py                 # builds data/rtl_bugs.csv (seeded)
python -m pytest tests                 # run before every commit
python -m repair.evaluate --no-model   # repair baseline
```

Training runs take hours on a CPU. Do not start one unless asked.

## Rules (from the README)

- Never commit API keys or passwords; they go in `.env`.
- Never commit model weights (`checkpoints/`) or the dataset (`data/`).
- Both models are scored on the same test split: `split == "test"` from `splits.py`.

## Working rules for agents

- Match the existing style: module docstring with usage, `argparse` CLIs run as `python -m pkg.module`, plain functions.
- Verilog reaches iverilog/yosys only as a file in a temp dir with a timeout, as `rtl_tools.py` does.
- Stay inside the files the task names; ask before touching `splits.py` or `make_dataset.py`.
- No new dependencies, broad rewrites or deleted files without asking.
- Report what you actually ran and its real output. Never invent numbers.

## Roles

| Role | Does | Claude | Codex |
|---|---|---|---|
| lead | plans, splits work into small tasks, verifies diffs and tests | main session | |
| implementer | writes code for one task | | `.codex/agents/implementer.md` |
| test-writer | adds/updates pytest tests | `.claude/agents/test-writer.md` | `.codex/agents/test-writer.md` |
| debugger | reproduces and fixes a failure | `.claude/agents/debugger.md` | `.codex/agents/debugger.md` |
| reviewer | read-only review of a diff | `.claude/agents/reviewer.md` | `.codex/agents/reviewer.md` |

Two agents never edit the same file at once. Hand-offs use `.agents/handoff_template.md`.
