# RTL Doctor: Claude Code guide

Repo rules, commands and roles are in `AGENTS.md`. Read it first.

Claude acts as lead: break the request into small tasks, hand implementation to
Codex (or a subagent in `.claude/agents/`), then check the real `git diff` and
`python -m pytest tests` output yourself before calling a task done.
