"""
Score the repair model on the test split.

    python -m repair.evaluate                  # baseline + the adapter in checkpoints/repair
    python -m repair.evaluate --no-model       # baseline only (needs no model, runs in a minute)
    python -m repair.evaluate --adapter none   # baseline + deepseek-coder without fine-tuning

Fix rate  = share of patches that compile (Icarus Verilog) AND are proven to
            behave like the original module (Yosys equivalence check).
Baseline  = return the buggy input unchanged.

Fix rate is measured on the "scorable" modules: the original compiles on its
own and can be equivalence-checked. The others (they include other files, use
modules defined elsewhere, or are testbenches without outputs) cannot be
judged by the tools; for those only `exact` (the patch has the same text as
the original) is reported. Whether a module is scorable depends on its
original code only, so it is the same for every model.

Results go to outputs/repair_eval/ (ignored by git).
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import rtl_tools
from repair.prompt import FORMATS, apply_answer, normalize
from splits import add_split

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data", default=ROOT / "data" / "rtl_bugs.csv", type=Path)
    p.add_argument("--adapter", default=str(ROOT / "checkpoints" / "repair"),
                   help="LoRA adapter folder, or 'none' for the base model without fine-tuning")
    p.add_argument("--no-model", action="store_true", help="only score the baseline")
    p.add_argument("--format", default=None, choices=FORMATS, help="answer format (default: the adapter's own)")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--limit", default=None, type=int, help="use only this many modules")
    p.add_argument("--batch-size", default=4, type=int)
    p.add_argument("--out", default=ROOT / "outputs" / "repair_eval", type=Path)
    return p.parse_args()


def is_scorable(original):
    """Can the tools judge a patch for this module? Depends on the original code only."""
    return rtl_tools.compiles(original)[0] and rtl_tools.checkable(original)


def score(original, patch, scorable):
    """Judge one patch (None = the model gave no usable answer)."""
    result = {"valid": patch is not None, "exact": False, "compiles": False, "fixed": False}
    if patch is None:
        return result
    if patch.split() == original.split():           # same text up to whitespace: no tools needed
        return {**result, "exact": True, "compiles": scorable, "fixed": scorable}
    if scorable:
        result["compiles"] = rtl_tools.compiles(patch)[0]
        result["fixed"] = (result["compiles"]
                           and rtl_tools.equivalent(original, patch)[0] == rtl_tools.EQUIVALENT)
    return result


def run_parallel(fn, items):
    with ThreadPoolExecutor(4) as pool:             # the tools are separate processes
        return list(pool.map(lambda args: fn(*args), items))


def summarize(rows, with_model):
    """Rates in percent, per bug type and overall."""
    def rates(group, scorable):
        out = {"n": len(group)}
        if with_model:
            out["valid"], out["exact"] = group.valid.mean(), group.exact.mean()
        out["scorable"] = len(scorable)
        if with_model:
            out["compiles"], out["fix_rate"] = scorable.compiles.mean(), scorable.fixed.mean()
        out["baseline_fix_rate"] = scorable.base_fixed.mean()
        return out
    table = {bug: rates(g, g[g.scorable]) for bug, g in rows.groupby("bug_type")}
    table["ALL"] = rates(rows, rows[rows.scorable])
    table = pd.DataFrame(table).T
    percent = [c for c in table.columns if c not in ("n", "scorable")]
    table[percent] = (100 * table[percent].astype(float)).round(1)
    return table.astype({"n": int, "scorable": int})


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = add_split(pd.read_csv(args.data))
    rows = df[(df.split == args.split) & (df.label == "buggy")]
    if args.limit:
        rows = rows.head(args.limit)
    originals = [normalize(o) for o in rows.original_code]
    buggy = [normalize(c) for c in rows.code]
    scorable = run_parallel(is_scorable, [(o,) for o in originals])
    print(f"{args.split} split: {len(rows)} buggy modules, {sum(scorable)} scorable")

    # baseline: hand the buggy module back unchanged
    base = run_parallel(score, zip(originals, buggy, scorable))
    rows = rows.assign(scorable=scorable, base_fixed=[s["fixed"] for s in base])
    results = {"split": args.split, "model": None}

    if not args.no_model:
        from repair.model import adapter_format, generate_answers, load
        adapter = None if args.adapter.lower() == "none" else args.adapter
        fmt = args.format or (adapter_format(adapter) if adapter else "edit")
        results["model"] = f"deepseek-coder + LoRA ({adapter})" if adapter else "deepseek-coder, no fine-tuning"
        results["format"] = fmt
        model, tok = load(adapter)
        start = time.time()

        def progress(i, n):
            print(f"  generated {i}/{n}  ({(time.time() - start) / 60:.1f} min)", flush=True)
        answers = generate_answers(model, tok, buggy, list(rows.bug_type), fmt, args.batch_size, progress)
        patches = [apply_answer(b, a, fmt) for b, a in zip(buggy, answers)]
        scores = run_parallel(score, zip(originals, patches, scorable))
        rows = rows.assign(**{k: [s[k] for s in scores] for k in scores[0]})

        with open(args.out / "predictions.jsonl", "w", encoding="utf-8") as f:
            for r, answer, patch in zip(rows.itertuples(), answers, patches):
                f.write(json.dumps({"module": r.module, "bug_type": r.bug_type, "bug_line": r.bug_line,
                                    "scorable": r.scorable, "valid": r.valid, "exact": r.exact,
                                    "compiles": r.compiles, "fixed": r.fixed,
                                    "baseline_fixed": r.base_fixed, "answer": answer, "patch": patch}) + "\n")

    table = summarize(rows, with_model=not args.no_model)
    results["table"] = table.to_dict("index")
    print(f"\n{results['model'] or 'Baseline only'}\n{table.to_string()}\n")
    if not args.no_model:
        print("valid: the answer could be applied   exact: the patch has the original's text   (both over all n modules)\n"
              "compiles, fix_rate: over the scorable modules; fix_rate = compiles and proven equivalent to the original")
    print("baseline_fix_rate: fix rate of returning the input unchanged (same scorable modules)")
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"saved {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
