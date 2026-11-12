"""
The train / val / test split, shared by the bug detector and the repair model.

A module's split depends only on its original (bug-free) code, so it is the
same on every machine, for every row order, and for any rebuild of the dataset.
Both models must be scored on split == "test".

    import pandas as pd
    from splits import add_split
    df = add_split(pd.read_csv("data/rtl_bugs.csv"))
    test = df[df.split == "test"]
"""
import hashlib
import re

TEST_PERCENT = 10
VAL_PERCENT = 10


def split_of(original_code):
    """'train', 'val' or 'test' for one module, from a hash of its original code."""
    key = re.sub(r"\s+", "", original_code)     # ignore whitespace / line-ending differences
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < TEST_PERCENT:
        return "test"
    if bucket < TEST_PERCENT + VAL_PERCENT:
        return "val"
    return "train"


def add_split(df):
    """Return a copy of the dataset with a `split` column."""
    df = df.copy()
    df["split"] = df["original_code"].map(split_of)
    return df
