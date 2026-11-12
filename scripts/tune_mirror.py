"""Tune the mirror clear-day check on training locations and training years only."""
import argparse

import numpy as np
import pandas as pd

from src.common import load_config
from src.detect import consistency as C


def train_frames(cfg):
    test_year = int(cfg["test_year"])
    frames = {}
    for loc in cfg["locations"]:
        if loc.get("split") != "train":
            continue
        df = pd.read_parquet(f"data/clean/{loc['id']}.parquet")
        df.index = pd.to_datetime(df.index)
        frames[loc["id"]] = df[df.index.year < test_year]
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=0.05,
                    help="max clean-data alarm episodes per location-year")
    ap.add_argument("--nb-clear-thr", type=float, default=C.DEFAULTS["nb_clear_thr"])
    ap.add_argument("--ref-mode", default="rolling", choices=["rolling", "seasonal"])
    a = ap.parse_args()

    cfg = load_config()
    frames = train_frames(cfg)
    base = C.merge_params({**(cfg.get("consistency") or {}), "mirror": True,
                           "nb_clear_thr": a.nb_clear_thr, "ref_mode": a.ref_mode})

    print("Per-site clear days and clear-day ratio noise (training years):")
    rows = []
    for name, df in frames.items():
        dev = C.mirror_deviation(df, base)
        v = dev[~np.isnan(dev)]
        rows.append({"site": name, "days": len(df),
                     "clear_share": round(len(v) / len(df), 3),
                     "dev_std": round(float(np.std(v)), 4) if v.size else np.nan,
                     "dev_p05": round(float(np.percentile(v, 5)), 4) if v.size else np.nan})
    print(pd.DataFrame(rows).to_string(index=False))

    table, best = [], None
    for n in (5, 10, 15, 20, 30):
        p = {**base, "under_n": n}
        rates = C.mirror_natural_alarms(frames.values(), p)
        tol = C.choose_under_tol(frames.values(), p, a.budget)
        ep = float(rates.loc[tol, "episodes_per_year"])
        safe = ep <= a.budget
        table.append({"under_n": n, "chosen_tol": tol, "episodes_per_year": round(ep, 3), "within_budget": safe})
        if safe and (best is None or tol < best[1] or (tol == best[1] and n < best[0])):
            best = (n, tol)

    print(f"\nSmallest safe tolerance per averaging length "
          f"(budget {a.budget} clean alarm episodes per location-year):")
    print(pd.DataFrame(table).to_string(index=False))

    if best is None:
        print("\nNo setting met the budget. Try --nb-clear-thr 0.75 or a larger --budget.")
        return
    n, tol = best
    print(f"\nRecommended: under_n={n}, under_tol={tol} "
          f"(flags downward shifts of about {tol * 100:.1f}% or more)")
    print("Add to config/tuned.yaml under consistency:")
    print(f"  mirror: true\n  under_n: {n}\n  under_tol: {tol}")


if __name__ == "__main__":
    main()