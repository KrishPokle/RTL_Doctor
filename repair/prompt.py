"""
What the repair model reads and what it writes.

Input  (both formats): the bug type plus the whole buggy module.
Output, format "edit":   one SEARCH/REPLACE block: the buggy lines, then the fixed lines.
Output, format "module": the whole corrected module.

Either way the caller ends up with a full patched module (apply_answer).
"edit" is the default: the answer is ~50 tokens instead of ~750, so training and
inference are much cheaper, and the model cannot corrupt lines it never touches.
"""
import difflib
import re

FORMATS = ("edit", "module")

BUG_HINTS = {
    "blocking_in_seq": "a non-blocking assignment (<=) in sequential logic was written as a blocking one (=)",
    "missing_default": "a `default:` branch was removed from a case statement",
    "width_off_by_one": "a bus is declared one bit too narrow",
    "inverted_condition": "an `if` condition is inverted",
    "reset_polarity": "a reset uses the wrong edge (posedge / negedge)",
    "wrong_operator": "an `assign` uses the wrong operator (& | + -)",
}

ASK = {
    "edit": ("Reply with one SEARCH/REPLACE block: the buggy lines copied exactly from the module, "
             "then the corrected lines."),
    "module": "Reply with the complete corrected module in a ```verilog block.",
}

_EDIT_RE = re.compile(r"<{7} SEARCH\n(.*?)\n={7}\n(.*?)\n?>{7} REPLACE", re.S)
_FENCE_RE = re.compile(r"```(?:verilog|systemverilog|v|sv)?[ \t]*\n(.*?)\n?```", re.S)


def normalize(code):
    """Unix line endings. Changes nothing about what the Verilog means."""
    return code.replace("\r\n", "\n").replace("\r", "\n")


def build_prompt(code, bug_type, fmt="edit"):
    """The user message for one buggy module."""
    hint = BUG_HINTS.get(bug_type)
    bug = f"{bug_type} ({hint})" if hint else bug_type
    return (f"Fix the bug in this Verilog module.\nBug type: {bug}\n\n"
            f"```verilog\n{normalize(code)}\n```\n\n{ASK[fmt]}")


# ------------------------------------------------------------------ edit format
def _find(lines, search):
    """Start indexes where `search` occurs in `lines`, ignoring indentation and trailing spaces."""
    want = [s.strip() for s in search]
    have = [l.strip() for l in lines]
    return [k for k in range(len(have) - len(want) + 1) if have[k:k + len(want)] == want]


def make_edit(buggy, original):
    """
    The smallest (search, replace) pair that turns `buggy` into `original`.
    Context lines are added until the search text is non-blank and unique in the module.
    Returns None when the two are identical.
    """
    b, o = normalize(buggy).split("\n"), normalize(original).split("\n")
    changed = [op for op in difflib.SequenceMatcher(None, b, o, autojunk=False).get_opcodes()
               if op[0] != "equal"]
    if not changed:
        return None
    i1, i2 = changed[0][1], changed[-1][2]      # changed span in the buggy lines
    j1, j2 = changed[0][3], changed[-1][4]      # the same span in the original lines
    lo, hi = i1, i2
    while True:
        search = b[lo:hi]
        if any(s.strip() for s in search) and len(_find(b, search)) == 1:
            break
        if lo == 0 and hi == len(b):
            break                                # whole file: cannot get more specific
        lo, hi = max(0, lo - 1), min(len(b), hi + 1)
    replace = o[j1 - (i1 - lo):j2 + (hi - i2)]
    return "\n".join(search), "\n".join(replace)


def format_edit(search, replace):
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"


def apply_edit(buggy, search, replace):
    """Replace the first place `search` occurs in `buggy`. None if it does not occur."""
    lines, want = normalize(buggy).split("\n"), normalize(search).split("\n")
    if not any(s.strip() for s in want):
        return None
    hits = _find(lines, want)
    if not hits:
        return None
    k = hits[0]
    new = normalize(replace).split("\n") if replace else []
    return "\n".join(lines[:k] + new + lines[k + len(want):])


# ------------------------------------------------------------------ both formats
def build_answer(buggy, original, fmt="edit"):
    """The answer the model is trained to give. None if this pair cannot be expressed."""
    if fmt == "module":
        answer = f"```verilog\n{normalize(original)}\n```"
    else:
        edit = make_edit(buggy, original)
        if edit is None:
            return None
        answer = format_edit(*edit)
    # keep only pairs whose answer really reproduces the original
    return answer if apply_answer(buggy, answer, fmt) == normalize(original) else None


def apply_answer(buggy, answer, fmt="edit"):
    """Turn the model's answer into a full patched module. None if the answer is unusable."""
    answer = normalize(answer)
    if fmt == "module":
        m = _FENCE_RE.search(answer)
        return m.group(1) if m else None
    m = _EDIT_RE.search(answer)
    return apply_edit(buggy, m.group(1), m.group(2)) if m else None
