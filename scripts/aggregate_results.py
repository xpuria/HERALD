"""
aggregate held-out test metrics across branches into one comparison table.
reads CSVs produced by each branch's training script and the baseline (main).
"""

import os
import sys
from io import StringIO
from subprocess import check_output

import pandas as pd

sys.path.append('.')


SOURCES = [
    ("main",                     "results/baseline_main.csv",  ["LSTM", "Transformer", "Fin-HRM (LoRA)", "Fin-HRM (Std)"]),
    ("feat/patch-embedding",     "results/patch_hrm.csv",      None),
    ("feat/multifeature",        "results/multifeature_hrm.csv", None),
    ("feat/aux-vol-smoothing",   "results/patch_hrm_aux.csv",  None),
    ("feat/recursive-trm",       "results/recursive_trm.csv",  None),
]


def read_from_branch(branch: str, path: str) -> pd.DataFrame:
    if branch == "main" and os.path.exists(path):
        return pd.read_csv(path)
    out = check_output(["git", "show", f"{branch}:{path}"]).decode()
    return pd.read_csv(StringIO(out))


def main():
    rows = []
    for branch, path, keep in SOURCES:
        df = read_from_branch(branch, path)
        if keep is not None:
            df = df[df["Model"].isin(keep)]
        df = df.copy()
        df["Branch"] = branch
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    cols = ["Branch", "Model", "Params", "TestN", "Loss", "Accuracy", "Precision", "Recall", "F1", "MCC"]
    out = out[cols]
    out_path = "results/final_comparison.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
