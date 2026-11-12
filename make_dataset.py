"""
Build the RTL bug dataset for the proposal.

Source: shailja/Verilog_GitHub (Hugging Face) -- open-source Verilog files.
Method: keep small single-module files, then inject ONE realistic bug into
most of them (the rest stay clean). Each row = one Verilog module.

Outputs (in ./data):
  rtl_bugs.csv               full dataset
  df_head.txt                df.head() for the proposal
  bug_type_distribution.png  the plot for the proposal
"""
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset

N_MODULES = 3000        # how many clean modules to collect
MAX_CHARS = 3000        # ~fits CodeT5+'s 512-token window
CLEAN_FRACTION = 0.20   # share of modules left bug-free
MAX_PER_BUG = 400       # cap per bug type so no class dominates
MAX_CLEAN = 900         # cap on clean rows
MAX_SCAN = 40000        # stop after this many usable files
SEED = 42
OUT = Path(__file__).parent / "data"

random.seed(SEED)


# ---------------------------------------------------------------- bug injectors
# Each takes code and returns (buggy_code, line_number) or None if not applicable.
# A pattern marks the text to change with the group (?P<fix>...). Only that text
# changes: the spacing around it stays as the author wrote it.

_NOT_CODE = re.compile(r'//[^\n]*|/\*.*?(?:\*/|\Z)|"(?:\\.|[^"\\\n])*"', re.S)   # comments and strings


def _blank_comments(code):
    """The code with comments and strings turned into spaces (same length, same lines)."""
    return _NOT_CODE.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), code)


def _line_of(code, idx):
    return code.count("\n", 0, idx) + 1


def _sub_one(code, pattern, repl, flags=re.M):
    # search the code without its comments, so a bug never lands in a comment or a string
    matches = list(re.finditer(pattern, _blank_comments(code), flags))
    if not matches:
        return None
    m = random.choice(matches)
    start, end = m.span("fix")
    new = repl(m) if callable(repl) else repl
    return code[:start] + new + code[end:], _line_of(code, start)


def blocking_in_seq(code):
    # q <= d;  ->  q = d;   (non-blocking turned blocking)
    return _sub_one(code, r"^\s*[A-Za-z_]\w*(?:\[[^\]]*\])?\s*(?P<fix><=)", "=")


def missing_default(code):
    # delete a one-line `default: ...;` from a case statement -> latch
    return _sub_one(code, r"(?P<fix>^[ \t]*default\s*:[^\n]*;[ \t]*\n)", "")


def width_off_by_one(code):
    # [7:0] -> [6:0]
    return _sub_one(code, r"\[(?P<fix>[1-9]\d*)\s*:\s*0\]", lambda m: str(int(m.group("fix")) - 1))


def inverted_condition(code):
    # if (en) -> if (!en)      (the empty group marks where the `!` goes)
    return _sub_one(code, r"\bif\s*\(\s*(?P<fix>)[A-Za-z_]\w*\s*\)", "!")


def reset_polarity(code):
    # negedge rst_n <-> posedge rst_n
    def flip(m):
        return "posedge" if m.group("fix") == "negedge" else "negedge"
    return _sub_one(code, r"\b(?P<fix>negedge|posedge)\s+\w*(?:rst|reset)\w*", flip, flags=re.M | re.I)


def wrong_operator(code):
    # a & b -> a | b,  a + b -> a - b  (inside assign statements)
    swaps = {"&": "|", "|": "&", "+": "-", "-": "+"}
    return _sub_one(code, r"\bassign\b[^;\n]*?\w\s*(?P<fix>[&|+-])(?![&|=])", lambda m: swaps[m.group("fix")])


INJECTORS = {
    "blocking_in_seq": blocking_in_seq,
    "missing_default": missing_default,
    "width_off_by_one": width_off_by_one,
    "inverted_condition": inverted_condition,
    "reset_polarity": reset_polarity,
    "wrong_operator": wrong_operator,
}


# ---------------------------------------------------------------- collect data
def is_usable(code):
    return (
        len(code) <= MAX_CHARS
        and code.count("endmodule") == 1
        and re.search(r"^\s*module\s+\w+", code, re.M)
    )


def module_name(code):
    m = re.search(r"^\s*module\s+(\w+)", code, re.M)
    return m.group(1) if m else "?"


def main():
    OUT.mkdir(exist_ok=True)
    ds = load_dataset("shailja/Verilog_GitHub", split="train", streaming=True)

    rows, seen, counts, scanned = [], set(), {}, 0
    for rec in ds:
        code = rec["text"]
        if not is_usable(code):
            continue
        key = re.sub(r"\s+", "", code)        # the source has duplicate files
        if key in seen:
            continue
        seen.add(key)

        row = {
            "module": module_name(code),
            "code": code,
            "label": "clean",
            "bug_type": "none",
            "bug_line": -1,
            "n_lines": code.count("\n") + 1,
            "original_code": code,
        }

        scanned += 1
        if scanned > MAX_SCAN:
            break
        if random.random() < CLEAN_FRACTION:
            if counts.get("none", 0) >= MAX_CLEAN:
                continue
            counts["none"] = counts.get("none", 0) + 1
        else:
            # try every bug type, keep the one we have the fewest of (keeps classes balanced)
            applicable = []
            for name, fn in INJECTORS.items():
                result = fn(code)
                if result and result[0] != code:
                    applicable.append((counts.get(name, 0), random.random(), name, result))
            applicable = [a for a in applicable if a[0] < MAX_PER_BUG]
            if not applicable:
                continue                      # no bug fits (or all full): skip it
            _, _, name, result = min(applicable)
            counts[name] = counts.get(name, 0) + 1
            row.update(code=result[0], label="buggy", bug_type=name, bug_line=result[1])
        rows.append(row)
        if len(rows) >= N_MODULES:
            break

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rtl_bugs.csv", index=False)

    # ---- df.head() for the proposal (code shortened so it fits on a slide)
    preview = df.drop(columns=["original_code"]).copy()
    preview["code"] = (preview["code"]
                       .str.replace(r"(?s)^.*?(?=\bmodule\b)", "", regex=True)  # skip header comments
                       .str.replace(r"\s+", " ", regex=True).str[:45] + "...")
    with pd.option_context("display.width", 200, "display.max_columns", 10):
        head = preview.head().to_string()
    (OUT / "df_head.txt").write_text(
        f"Dataset: {len(df):,} rows x {df.shape[1]} columns\n\n{head}\n", encoding="utf-8")
    print(head)
    print(f"\n{len(df):,} rows; label counts:\n{df['label'].value_counts()}")

    plot_bug_types(df)


def plot_bug_types(df):
    counts = df["bug_type"].value_counts().sort_values()
    labels = [("clean (no bug)" if k == "none" else k.replace("_", " ")) for k in counts.index]
    colors = ["#b5b4ad" if k == "none" else "#2a78d6" for k in counts.index]

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    bars = ax.barh(labels, counts.values, color=colors, height=0.6)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_width() + counts.max() * 0.01, b.get_y() + b.get_height() / 2,
                f"{v:,}", va="center", fontsize=9, color="#52514e")

    ax.set_title(f"Rows per bug type ({len(df):,} Verilog modules)",
                 loc="left", fontsize=12, color="#0b0b0b", pad=12)
    ax.set_xlabel("Number of modules", color="#52514e")
    ax.tick_params(colors="#52514e", length=0)
    ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "bug_type_distribution.png", facecolor=fig.get_facecolor())
    print(f"saved {OUT / 'bug_type_distribution.png'}")


if __name__ == "__main__":
    main()
