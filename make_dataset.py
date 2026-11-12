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

def _line_of(code, idx):
    return code.count("\n", 0, idx) + 1


def _sub_one(code, pattern, repl, flags=re.M):
    matches = list(re.finditer(pattern, code, flags))
    if not matches:
        return None
    m = random.choice(matches)
    new = m.expand(repl) if isinstance(repl, str) else repl(m)
    return code[:m.start()] + new + code[m.end():], _line_of(code, m.start())


def blocking_in_seq(code):
    # q <= d;  ->  q = d;   (non-blocking turned blocking)
    return _sub_one(code, r"^(\s*[A-Za-z_]\w*(?:\[[^\]]*\])?\s*)<=", r"\1=")


def missing_default(code):
    # delete a one-line `default: ...;` from a case statement -> latch
    return _sub_one(code, r"^[ \t]*default\s*:[^\n]*;[ \t]*\n", "")


def width_off_by_one(code):
    # [7:0] -> [6:0]
    def shrink(m):
        return f"[{int(m.group(1)) - 1}:0]"
    return _sub_one(code, r"\[([1-9]\d*)\s*:\s*0\]", shrink)


def inverted_condition(code):
    # if (en) -> if (!en)
    return _sub_one(code, r"\bif\s*\(\s*([A-Za-z_]\w*)\s*\)", r"if (!\1)")


def reset_polarity(code):
    # negedge rst_n <-> posedge rst_n
    def flip(m):
        edge = "posedge" if m.group(1) == "negedge" else "negedge"
        return f"{edge} {m.group(2)}"
    return _sub_one(code, r"\b(negedge|posedge)\s+(\w*(?:rst|reset)\w*)", flip, flags=re.M | re.I)


def wrong_operator(code):
    # a & b -> a | b,  a + b -> a - b  (inside assign statements)
    swaps = {"&": "|", "|": "&", "+": "-", "-": "+"}
    def swap(m):
        return f"{m.group(1)} {swaps[m.group(2)]} "
    return _sub_one(code, r"(\bassign\b[^;\n]*?\w)\s*([&|+-])(?![&|=])\s*", swap)


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
