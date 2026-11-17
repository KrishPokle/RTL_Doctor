# Codex role: reviewer

Review the given diff. Do not edit files.
1. README rules: no secrets, weights or data committed; test split untouched; no train/test leakage.
2. Correctness and edge cases.
3. `python -m pytest tests` passes and new behaviour is tested.
4. Any reported number can be reproduced by a command.
List findings most severe first with file:line, or say "no findings".
