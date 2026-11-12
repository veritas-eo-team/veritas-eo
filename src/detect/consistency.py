"""Model-free physical consistency checks (contract C4).

over_clearsky:  target > ctx_CLRSKY_SFC_SW_DWN * (1 + clearsky_tol)
                (all-sky radiation cannot exceed clear-sky; skipped when the column is
                absent or NaN)
cloud_conflict: on Sentinel-2 days with cloud >= cloud_thr, clearness
                (target / clear-sky) >= clearness_thr (no sunny readings under heavy cloud)
under_clearsky (mirror check, off unless `mirror` is true):
                on Sentinel-2 days with cloud <= clear_cloud_thr, clearness is below the
                site's own reference clear-day clearness * (1 - under_tol). The reference
                is the median clear-day clearness from `ref_span` to `ref_lag` days ago,
                so it is causal and a fresh attack cannot immediately bias it.

The checks have separate thresholds: a Sentinel-2 snapshot (one ~3 km box at
~10:30) and a NASA daily value (a ~50 km cell, whole day) disagree now and then
naturally, so the cloud checks need more evidence than the clear-sky check.
score = max(over-clear-sky days in the last `window` days / min_violations,
            cloud conflicts in the last `window` days / cloud_min_violations,
            under-clear-sky days in the last `under_window` days / under_min_violations)
alarm = score >= 1. Causal; missing columns give score 0.
natural_exceedance_rate(), cloud_conflict_rate() and under_natural_rate() measure how
often clean data break each check; tuning uses them.
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
    "cloud_min_violations": 4,
    # mirror check (downward bias on clear days)
    "mirror": False,
    "clear_cloud_thr": 0.1,
    "under_tol": 0.05,
    "under_window": 30,
    "under_min_violations": 2,
    "ref_lag": 60,
    "ref_span": 365,
    "ref_min_days": 8,
}
TOL_GRID: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15)
UNDER_TOL_GRID: tuple[float, ...] = (0.02, 0.03, 0.04, 0.05, 0.075, 0.1)

_INT_KEYS = ("window", "min_violations", "cloud_min_violations",
             "under_window", "under_min_violations", "ref_lag", "ref_span", "ref_min_days")
_FLOAT_KEYS = ("clearsky_tol", "cloud_thr", "clearness_thr", "clear_cloud_thr", "under_tol")


# ------------------------------------------------------------------ params

def merge_params(params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """DEFAULTS overridden by non-None values in `params`; validated."""
    merged = dict(DEFAULTS)
    for key, value in (params or {}).items():
        if value is not None:
            merged[key] = value
    for key in _INT_KEYS:
        merged[key] = int(merged[key])
    for key in _FLOAT_KEYS:
        merged[key] = float(merged[key])
    merged["mirror"] = bool(merged["mirror"])
    if merged["window"] < 1 or merged["min_violations"] < 1 or merged["cloud_min_violations"] < 1:
        raise ValueError("consistency.window, min_violations and cloud_min_violations must be at least 1")
    if merged["under_window"] < 1 or merged["under_min_violations"] < 1 or merged["ref_min_days"] < 1:
        raise ValueError("consistency.under_window, under_min_violations and ref_min_days must be at least 1")
    if merged["ref_lag"] < 0 or merged["ref_span"] <= merged["ref_lag"]:
        raise ValueError("consistency.ref_span must be greater than ref_lag (and ref_lag >= 0)")
    if merged["clearsky_tol"] < 0:
        raise ValueError("consistency.clearsky_tol must be >= 0")
    if not 0.0 < merged["cloud_thr"] <= 1.0:
        raise ValueError("consistency.cloud_thr must be in (0, 1]")
    if not 0.0 <= merged["clear_cloud_thr"] < merged["cloud_thr"]:
        raise ValueError("consistency.clear_cloud_thr must be in [0, cloud_thr)")
    if merged["clearness_thr"] <= 0:
        raise ValueError("consistency.clearness_thr must be positive")
    if not 0.0 < merged["under_tol"] < 1.0:
        raise ValueError("consistency.under_tol must be in (0, 1)")
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


def clear_day_clearness(df: pd.DataFrame, clear_cloud_thr: float) -> np.ndarray:
    """target / clear-sky on Sentinel-2 days with cloud <= clear_cloud_thr, NaN elsewhere."""
    target = _column(df, "target")
    clr = _column(df, CLRSKY_COL)
    cloud = _column(df, "cloud")
    with np.errstate(invalid="ignore", divide="ignore"):
        clearness = np.where(clr > 0, target / clr, np.nan)
        clear = cloud <= clear_cloud_thr
    return np.where(clear, clearness, np.nan)


def reference_clearness(clear_cl: np.ndarray, ref_lag: int, ref_span: int, ref_min_days: int) -> np.ndarray:
    """Causal reference: median clear-day clearness from ref_span to ref_lag days ago.

    Assumes one row per day (the aligned frames are daily). NaN until ref_min_days
    clear days are available in the reference window.
    """
    s = pd.Series(clear_cl)
    return s.shift(ref_lag).rolling(ref_span - ref_lag, min_periods=ref_min_days).median().to_numpy()


def under_clearsky(df: pd.DataFrame, p: Mapping[str, Any]) -> np.ndarray:
    """Mirror check flags: a clear day noticeably darker than the site's own clear days."""
    cl = clear_day_clearness(df, p["clear_cloud_thr"])
    ref = reference_clearness(cl, p["ref_lag"], p["ref_span"], p["ref_min_days"])
    with np.errstate(invalid="ignore"):
        under = cl < ref * (1.0 - p["under_tol"])
    return under & ~np.isnan(cl) & ~np.isnan(ref)


def _rolling_count(flags: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(flags.astype(float)).rolling(window, min_periods=1).sum().to_numpy()


def score_from(over: np.ndarray, conflict: np.ndarray, p: Mapping[str, Any],
               under: np.ndarray | None = None) -> np.ndarray:
    """max of each check's violation count / its required count over its window."""
    score = np.maximum(_rolling_count(over, p["window"]) / p["min_violations"],
                       _rolling_count(conflict, p["window"]) / p["cloud_min_violations"])
    if under is not None:
        score = np.maximum(score, _rolling_count(under, p["under_window"]) / p["under_min_violations"])
    return score


def detect(df: pd.DataFrame, gap: pd.Series | None = None, params: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Contract C4: DataFrame[score, alarm, over_clearsky, cloud_conflict(, under_clearsky)] on df.index.

    `gap` is accepted for the common detector signature and not used (the checks are
    model-free). The under_clearsky column is present only when the mirror check is on.
    attrs: est_start (first violation in the window that triggered the first alarm,
    Timestamp or None), first_alarm.
    """
    p = merge_params(params)
    over, conflict = violations(df, p["clearsky_tol"], p["cloud_thr"], p["clearness_thr"])
    under = under_clearsky(df, p) if p["mirror"] else None
    score = score_from(over, conflict, p, under)
    alarm = score >= 1.0
    cols = {"score": score, "alarm": alarm, "over_clearsky": over, "cloud_conflict": conflict}
    if under is not None:
        cols["under_clearsky"] = under
    out = pd.DataFrame(cols, index=df.index)

    est_start = first_alarm = None
    hits = np.flatnonzero(alarm)
    if hits.size:
        a = int(hits[0])
        span = max(p["window"], p["under_window"]) if under is not None else p["window"]
        lo = max(0, a - span + 1)
        bad = over | conflict | (under if under is not None else False)
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


def cloud_conflict_rate(frames: Iterable[pd.DataFrame], cloud_thr: float = DEFAULTS["cloud_thr"],
                        clearness_thr: float = DEFAULTS["clearness_thr"]) -> dict[str, float]:
    """On clean data: share of heavy-cloud Sentinel-2 days that still look clear in NASA POWER."""
    cloudy = conflicts = 0
    for df in frames:
        cloud = _column(df, "cloud")
        _, conflict = violations(df, 0.0, cloud_thr, clearness_thr)
        cloudy += int(np.sum(cloud >= cloud_thr))
        conflicts += int(np.sum(conflict))
    return {"cloudy_days": cloudy, "conflict_days": conflicts,
            "rate": conflicts / cloudy if cloudy else float("nan")}


def under_natural_rate(frames: Iterable[pd.DataFrame], params: Mapping[str, Any] | None = None,
                       tol_grid: Sequence[float] = UNDER_TOL_GRID) -> pd.DataFrame:
    """On clean data: share of clear days (with a reference) flagged by the mirror check, per under_tol.

    Returns a DataFrame indexed by under_tol with columns rate, flagged_days, clear_days.
    """
    frames = list(frames)
    base = merge_params(params)
    rows = []
    for tol in tol_grid:
        p = {**base, "under_tol": float(tol)}
        flagged = clear_days = 0
        for df in frames:
            cl = clear_day_clearness(df, p["clear_cloud_thr"])
            ref = reference_clearness(cl, p["ref_lag"], p["ref_span"], p["ref_min_days"])
            valid = ~np.isnan(cl) & ~np.isnan(ref)
            clear_days += int(valid.sum())
            flagged += int(under_clearsky(df, p).sum())
        rows.append({"under_tol": float(tol), "rate": flagged / clear_days if clear_days else float("nan"),
                     "flagged_days": flagged, "clear_days": clear_days})
    return pd.DataFrame(rows).set_index("under_tol")


def choose_under_tol(frames: Iterable[pd.DataFrame], params: Mapping[str, Any] | None = None,
                     max_rate: float = 0.02, tol_grid: Sequence[float] = UNDER_TOL_GRID) -> float:
    """Smallest under_tol whose natural flag rate on clean clear days is <= max_rate."""
    rates = under_natural_rate(frames, params, tol_grid)
    ok = rates.index[rates["rate"] <= max_rate]
    return float(ok.min()) if len(ok) else float(max(tol_grid))