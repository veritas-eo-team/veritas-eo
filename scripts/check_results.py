"""Compare a fresh benchmark table with a stored reference table."""
import argparse
import sys

import numpy as np
import pandas as pd

SKIP = ("ms", "time", "sec", "runtime", "mb", "memory", "peak")


def to_num(s):
    return pd.to_numeric(s.replace({"None": np.nan, "": np.nan}), errors="coerce")


def is_numeric(s):
    vals = s.replace({"None": np.nan, "": np.nan})
    num = to_num(s)
    return vals.notna().sum() > 0 and num.notna().sum() == vals.notna().sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", default="results/benchmark_table.csv")
    ap.add_argument("--ref", default="results/phase1_baseline/benchmark_table.csv")
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()

    new, ref = pd.read_csv(a.new, dtype=str), pd.read_csv(a.ref, dtype=str)

    if list(new.columns) != list(ref.columns):
        print("[FAIL] Columns differ")
        print("  new:", list(new.columns))
        print("  ref:", list(ref.columns))
        return 1
    if len(new) != len(ref):
        print(f"[FAIL] Row count differs: new {len(new)} vs ref {len(ref)}")
        return 1

    numeric = [c for c in ref.columns if is_numeric(ref[c])]
    keys = [c for c in ref.columns if c not in numeric]
    if keys:
        new = new.sort_values(keys).reset_index(drop=True)
        ref = ref.sort_values(keys).reset_index(drop=True)

    bad = []
    for c in ref.columns:
        if any(s in c.lower() for s in SKIP):
            continue
        if c in numeric:
            x, y = to_num(new[c]).to_numpy(float), to_num(ref[c]).to_numpy(float)
            same = np.isclose(x, y, atol=a.tol, rtol=0) | (np.isnan(x) & np.isnan(y))
        else:
            same = (new[c].fillna("") == ref[c].fillna("")).to_numpy()
        for i in np.flatnonzero(~same):
            row = " / ".join(str(ref.loc[i, k]) for k in keys) or f"row {i}"
            bad.append(f"{row}: {c}  new={new.loc[i, c]}  ref={ref.loc[i, c]}")

    if bad:
        print(f"[FAIL] {len(bad)} value(s) differ from the reference:")
        for b in bad[:50]:
            print("  ", b)
        return 1

    print(f"[OK] Benchmark matches the reference ({len(ref)} rows, {len(ref.columns)} columns).")
    return 0


if __name__ == "__main__":
    sys.exit(main())