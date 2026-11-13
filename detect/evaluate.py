"""
Score the bug detector on the test split, next to two baselines.

    python -m detect.evaluate               # baselines + the detector in checkpoints/detect
    python -m detect.evaluate --no-model    # baselines only (no model, runs in seconds)

Metric: macro-F1 over the 7 classes (the 6 bug types and "none"), so every bug
type counts the same however common it is. Also reported: accuracy, and F1 of
the clean/buggy decision alone.

Baselines:
  majority  always predict the most common class of the train split
  tfidf     TF-IDF of character 3-5-grams + logistic regression, trained on the
            train split: a classic model with no code pre-training

Results go to outputs/detect_eval/ (ignored by git).
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from detect.model import CHECKPOINT, LABELS
from splits import add_split

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data", default=ROOT / "data" / "rtl_bugs.csv", type=Path)
    p.add_argument("--checkpoint", default=CHECKPOINT, type=Path)
    p.add_argument("--no-model", action="store_true", help="only score the baselines")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--out", default=ROOT / "outputs" / "detect_eval", type=Path)
    return p.parse_args()


def scores(truth, pred):
    is_bug = lambda xs: [x != "none" for x in xs]
    return {"macro_f1": round(f1_score(truth, pred, labels=LABELS, average="macro", zero_division=0), 4),
            "accuracy": round(accuracy_score(truth, pred), 4),
            "buggy_f1": round(f1_score(is_bug(truth), is_bug(pred), zero_division=0), 4)}


def majority_baseline(train, test):
    return [train.bug_type.mode()[0]] * len(test)


def tfidf_baseline(train, test):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, C=10)
    clf.fit(vec.fit_transform(train.code), train.bug_type)
    return list(clf.predict(vec.transform(test.code)))


def main():
    args = parse_args()
    df = add_split(pd.read_csv(args.data))
    train, test = df[df.split == "train"], df[df.split == args.split]
    truth = list(test.bug_type)
    preds = {"majority": majority_baseline(train, test), "tfidf": tfidf_baseline(train, test)}
    if not args.no_model:
        from detect.model import load_detector, predict_many
        model, tok = load_detector(args.checkpoint)
        preds["codet5p"] = [p["bug_type"] for p in predict_many(model, tok, list(test.code), args.batch_size)]

    results = {name: scores(truth, p) for name, p in preds.items()}
    per_type = {name: {t: round(f1_score([x == t for x in truth], [x == t for x in p], zero_division=0), 4)
                       for t in LABELS} for name, p in preds.items()}

    print(f"{args.split} split: {len(test)} modules\n")
    print(pd.DataFrame(results).T.to_string(), "\n")
    print("F1 per class:")
    print(pd.DataFrame(per_type).to_string(), "\n")
    best = "codet5p" if "codet5p" in preds else "tfidf"
    print(f"{best}:\n" + classification_report(truth, preds[best], labels=LABELS, zero_division=0))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.split}_scores.json").write_text(json.dumps({"overall": results, "per_type": per_type}, indent=2))
    pd.DataFrame({"module": test.module, "truth": truth, **preds}).to_csv(args.out / f"{args.split}_predictions.csv",
                                                                          index=False)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
