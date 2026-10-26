"""Model-free physical consistency checks (contract C4).

over_clearsky:  target > ctx_CLRSKY_SFC_SW_DWN * (1 + clearsky_tol)
                (all-sky radiation cannot exceed clear-sky; skipped when the column is
                absent or NaN)
cloud_conflict: on Sentinel-2 days with cloud >= cloud_thr, clearness
                (target / clear-sky) >= clearness_thr (no sunny readings under heavy cloud)

score = violations in the last `window` days / min_violations; alarm = score >= 1.
Causal; missing columns give score 0. NASA's all-sky value can slightly exceed clear-sky
naturally, so natural_exceedance_rate() measures how often clean data break each
tolerance and tuning picks clearsky_tol from it.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.common import get_logger, load_config

log = get_logger(__name__)

CLRSKY_COL = "ctx_CLRSKY_SFC_SW_DWN"
DEFAULTS: dict[str, Any] = {
    "clearsky_tol": 0.03,
    "cloud_thr": 0.7,
    "clearness_thr": 0.8,
    "window": 14,
    "min_violations": 2,
}
TOL_GRID: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15)


# ------------------------------------------------------------------ params

def merge_params(params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """DEFAULTS overridden by non-None values in `params`; validated."""
    merged = dict(DEFAULTS)
    for key, value in (params or {}).items():
        if value is not None:
            merged[key] = value
    merged["window"] = int(merged["window"])
    merged["min_violations"] = int(merged["min_violations"])
    for key in ("clearsky_tol", "cloud_thr", "clearness_thr"):
        merged[key] = float(merged[key])
    if merged["window"] < 1 or merged["min_violations"] < 1:
        raise ValueError("consistency.window and min_violations must be at least 1")
    if merged["clearsky_tol"] < 0:
        raise ValueError("consistency.clearsky_tol must be >= 0")
    if not 0.0 < merged["cloud_thr"] <= 1.0:
        raise ValueError("consistency.cloud_thr must be in (0, 1]")
    if merged["clearness_thr"] <= 0:
        raise ValueError("consistency.clearness_thr must be positive")
    return merged


def params_from_config(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Consistency parameters from the config `consistency` section (tuned.yaml included)."""
    cfg = cfg if cfg is not None else load_config()
    return merge_params(cfg.get("consistency"))


# ------------------------------------------------------------------ checks

def _column(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[name].to_numpy(dtype=float) if name in df else np.full(len(df), np.nan)


def violations(df: pd.DataFrame, clearsky_tol: float, cloud_thr: float, clearness_thr: float) -> tuple[np.ndarray, np.ndarray]:
    """(over_clearsky, cloud_conflict) boolean arrays; NaN or missing inputs give False."""
    target = _column(df, "target")
    clr = _column(df, CLRSKY_COL)
    cloud = _column(df, "cloud")
    with np.errstate(invalid="ignore", divide="ignore"):
        over = target > clr * (1.0 + clearsky_tol)
        clearness = np.where(clr > 0, target / clr, np.nan)
        conflict = (cloud >= cloud_thr) & (clearness >= clearness_thr)
    return over, conflict


def detect(df: pd.DataFrame, gap: pd.Series | None = None, params: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Contract C4: DataFrame[score, alarm, over_clearsky, cloud_conflict] on df.index.

    `gap` is accepted for the common detector signature and not used (the checks are
    model-free). attrs: est_start (first violation in the window that triggered the first
    alarm, Timestamp or None), first_alarm.
    """
    p = merge_params(params)
    over, conflict = violations(df, p["clearsky_tol"], p["cloud_thr"], p["clearness_thr"])
    bad = over | conflict
    count = pd.Series(bad.astype(float)).rolling(p["window"], min_periods=1).sum().to_numpy()
    score = count / p["min_violations"]
    alarm = score >= 1.0
    out = pd.DataFrame({"score": score, "alarm": alarm, "over_clearsky": over, "cloud_conflict": conflict},
                       index=df.index)

    est_start = first_alarm = None
    hits = np.flatnonzero(alarm)
    if hits.size:
        a = int(hits[0])
        lo = max(0, a - p["window"] + 1)
        first = lo + int(np.flatnonzero(bad[lo:a + 1])[0])
        est_start, first_alarm = df.index[first], df.index[a]
    out.attrs.update(est_start=est_start, first_alarm=first_alarm)
    return out


# ------------------------------------------------------------------ tuning helpers

def natural_exceedance_rate(frames: Iterable[pd.DataFrame], tol_grid: Sequence[float] = TOL_GRID) -> pd.DataFrame:
    """Share of valid days where clean target > clear-sky * (1 + tol), pooled over frames.

    Returns a DataFrame indexed by tol with columns rate, exceed_days, n_days.
    """
    targets, clears = [], []
    for df in frames:
        t, c = _column(df, "target"), _column(df, CLRSKY_COL)
        ok = ~np.isnan(t) & ~np.isnan(c) & (c > 0)
        targets.append(t[ok])
        clears.append(c[ok])
    t = np.concatenate(targets) if targets else np.zeros(0)
    c = np.concatenate(clears) if clears else np.zeros(0)
    rows = []
    for tol in tol_grid:
        exceed = int(np.sum(t > c * (1.0 + tol)))
        rows.append({"tol": float(tol), "rate": exceed / len(t) if len(t) else 0.0,
                     "exceed_days": exceed, "n_days": int(len(t))})
    return pd.DataFrame(rows).set_index("tol")


def choose_clearsky_tol(frames: Iterable[pd.DataFrame], max_rate: float = 0.005,
                        tol_grid: Sequence[float] = TOL_GRID) -> float:
    """Smallest tolerance on the grid whose natural exceedance rate is <= max_rate."""
    rates = natural_exceedance_rate(frames, tol_grid)
    ok = rates.index[rates["rate"] <= max_rate]
    return float(ok.min()) if len(ok) else float(max(tol_grid))