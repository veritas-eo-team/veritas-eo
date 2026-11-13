"""Measure whether a long-memory (months-long) gap average could catch tiny attacks.

Training locations, training years only. Reports (1) the noise of the L-day mean gap
on clean data and (2) how much of a small step is still visible in the gap after the
per-site calibration has had time to adapt.
"""
import argparse

import numpy as np
import pandas as pd

from src.common import load_config
from src.data.align import load
from src.detect.context_model import ContextModel, calibrate_site
from src.pipeline import _ref_params

WINDOWS = (30, 60, 90, 180, 365)
LAGS = (14, 30, 90, 180, 300)


def gap_of(df, model, ref_days, min_ref):
    return calibrate_site(model.predict(df), ref_days, min_ref)["gap"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=float, default=0.027)
    ap.add_argument("--attack-start", default="2022-03-01")
    a = ap.parse_args()

    cfg = load_config()
    model = ContextModel.load()
    ref_days, min_ref = _ref_params(cfg)
    test_year = int(cfg["test_year"])
    start = pd.Timestamp(a.attack_start)
    injected = float(np.log1p(a.size))

    pooled = {L: [] for L in WINDOWS}
    kept = {lag: [] for lag in LAGS}
    per_site = []

    for loc in cfg["locations"]:
        if loc.get("split") != "train":
            continue
        df = load(loc["id"], cfg)
        df = df[df.index.year < test_year]
        g = gap_of(df, model, ref_days, min_ref)
        warm = g.index[0] + pd.Timedelta(days=ref_days)

        row = {"site": loc["id"]}
        for L in WINDOWS:
            m = g.rolling(L, min_periods=int(0.8 * L)).mean()
            m = m[m.index >= warm + pd.Timedelta(days=L)].dropna()
            pooled[L].append(m.to_numpy())
            row[f"std_{L}d"] = round(float(m.std()), 4)
        per_site.append(row)

        att = df.copy()
        att.loc[att.index >= start, "target"] *= 1.0 + a.size
        diff = gap_of(att, model, ref_days, min_ref) - g
        for lag in LAGS:
            day = start + pd.Timedelta(days=lag)
            w = diff.loc[day - pd.Timedelta(days=6): day].dropna()
            if len(w):
                kept[lag].append(float(w.mean()))

    print(f"Calibration reference: ref_days={ref_days}, min_ref_days={min_ref}")
    print("\n1) Noise of the L-day mean gap on clean data, per training site (std):")
    print(pd.DataFrame(per_site).to_string(index=False))

    rows = []
    for L in WINDOWS:
        v = np.concatenate(pooled[L]) if pooled[L] else np.zeros(0)
        rows.append({"window_days": L, "std": round(float(np.std(v)), 4),
                     "p99.5_abs": round(float(np.percentile(np.abs(v), 99.5)), 4) if v.size else np.nan})
    noise = pd.DataFrame(rows)
    print("\n   Pooled over training sites:")
    print(noise.to_string(index=False))

    print(f"\n2) How much of a +{a.size:.1%} step is still in the gap (injected = {injected:.4f}):")
    krows = [{"days_after_start": lag,
              "mean_shift": round(float(np.mean(kept[lag])), 4) if kept[lag] else np.nan,
              "share_kept": f"{np.mean(kept[lag]) / injected:.0%}" if kept[lag] else "-"} for lag in LAGS]
    print(pd.DataFrame(krows).to_string(index=False))

    print("\n3) Signal-to-noise (shift kept after L days / p99.5 noise of the L-day mean):")
    for L in (30, 90, 180):
        lag = min(LAGS, key=lambda x: abs(x - L))
        if kept[lag]:
            snr = np.mean(kept[lag]) / float(noise.loc[noise.window_days == L, "p99.5_abs"].iloc[0])
            print(f"   L={L:>3}d: {snr:.2f}  ({'promising' if snr >= 1.5 else 'marginal' if snr >= 1.0 else 'not enough'})")


if __name__ == "__main__":
    main()