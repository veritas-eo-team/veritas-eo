"""Model-free physical consistency checks (contract C4).

over_clearsky:  target > ctx_CLRSKY_SFC_SW_DWN * (1 + clearsky_tol)
                (all-sky radiation cannot exceed clear-sky; skipped when the column is
                absent or NaN)
cloud_conflict: on Sentinel-2 days with cloud >= cloud_thr, clearness
                (target / clear-sky) >= clearness_thr (no sunny readings under heavy cloud)
under_clearsky (mirror check, off unless `mirror` is true): downward bias on clear days.
                A day is clear when every neighbour (nb_*) reaches nb_clear_thr of the
                clear-sky value and Sentinel-2, when present, is not cloudier than
                clear_cloud_thr. The attacker controls only the target, so the
                neighbours decide. On clear days r = target / mean(neighbours) is
                compared with the site's own reference ratio (median over the clear
                days from ref_span to ref_lag days ago, so it is causal). The mean
                relative deviation over the last under_n clear days drives the score.

score = max(over-clear-sky days in the last `window` days / min_violations,
            cloud conflicts in the last `window` days / cloud_min_violations,
            -mean deviation over the last under_n clear days / under_tol)
alarm = score >= 1. Causal; missing columns give score 0.
natural_exceedance_rate(), cloud_conflict_rate() and mirror_natural_alarms() measure
how often clean data break each check; tuning uses them.
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
    # mirror check (downward bias on clear days, neighbour-relative)
    "mirror": False,
    "nb_clear_thr": 0.7,
    "clear_cloud_thr": 0.3,
    "under_n": 10,
    "under_tol": 0.04,
    "under_stale": 20,
    "ref_lag": 60,
    "ref_span": 365,
    "ref_min_days": 20,
}
TOL_GRID: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15)
UNDER_TOL_GRID: tuple[float, ...] = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075)

_INT_KEYS = ("window", "min_violations", "cloud_min_violations",
             "under_n", "under_stale", "ref_lag", "ref_span", "ref_min_days")
_FLOAT_KEYS = ("clearsky_tol", "cloud_thr", "clearness_thr",
               "nb_clear_thr", "clear_cloud_thr", "under_tol")


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
    if merged["under_n"] < 1 or merged["under_stale"] < 0 or merged["ref_min_days"] < 1:
        raise ValueError("consistency.under_n and ref_min_days must be >= 1, under_stale >= 0")
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
    if not 0.0 < merged["nb_clear_thr"] <= 1.5:
        raise ValueError("consistency.nb_clear_thr must be in (0, 1.5]")
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


def _neighbours(df: pd.DataFrame) -> np.ndarray:
    cols = sorted(c for c in df.columns if str(c).startswith("nb_"))
    if not cols:
        return np.full((len(df), 0), np.nan)
    return df[cols].to_numpy(dtype=float)


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


def clear_days(df: pd.DataFrame, nb_clear_thr: float, clear_cloud_thr: float) -> np.ndarray:
    """Days every neighbour is near clear-sky and Sentinel-2 (if present) is not cloudy.

    Uses only neighbours, clear-sky and Sentinel-2, never the target itself.
    """
    nb = _neighbours(df)
    clr = _column(df, CLRSKY_COL)
    cloud = _column(df, "cloud")
    if nb.shape[1] == 0:
        return np.zeros(len(df), dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        nb_cl = nb / clr[:, None]
        nb_clear = np.all(nb_cl >= nb_clear_thr, axis=1) & (clr > 0)
        s2_cloudy = cloud > clear_cloud_thr
    return nb_clear & ~s2_cloudy


def _reference(series: np.ndarray, ref_lag: int, ref_span: int, ref_min_days: int) -> np.ndarray:
    """Causal reference: median of `series` from ref_span to ref_lag rows ago (one row per day)."""
    s = pd.Series(series)
    return s.shift(ref_lag).rolling(ref_span - ref_lag, min_periods=ref_min_days).median().to_numpy()


def mirror_deviation(df: pd.DataFrame, p: Mapping[str, Any]) -> np.ndarray:
    """Relative deviation r / ref - 1 of target / mean(neighbours) on clear days, NaN elsewhere."""
    clear = clear_days(df, p["nb_clear_thr"], p["clear_cloud_thr"])
    nb = _neighbours(df)
    target = _column(df, "target")
    if nb.shape[1] == 0:
        return np.full(len(df), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        nb_mean = np.nanmean(nb, axis=1) if nb.size else np.full(len(df), np.nan)
        ratio = np.where(clear & (nb_mean > 0), target / nb_mean, np.nan)
    ref = _reference(ratio, p["ref_lag"], p["ref_span"], p["ref_min_days"])
    with np.errstate(invalid="ignore", divide="ignore"):
        return ratio / ref - 1.0


def mirror_score(dev: np.ndarray, p: Mapping[str, Any]) -> np.ndarray:
    """max(0, -mean deviation over the last under_n clear days) / under_tol, held for under_stale days."""
    s = pd.Series(dev)
    valid = s.dropna()
    if valid.empty:
        return np.zeros(len(dev))
    mean_dev = valid.rolling(p["under_n"], min_periods=p["under_n"]).mean()
    held = mean_dev.reindex(s.index).ffill(limit=p["under_stale"]) if p["under_stale"] else mean_dev.reindex(s.index)
    score = np.clip(-held.to_numpy(), 0.0, None) / p["under_tol"]
    return np.nan_to_num(score, nan=0.0)


def under_clearsky(df: pd.DataFrame, p: Mapping[str, Any]) -> np.ndarray:
    """Per-day flags: a clear day darker than its reference by more than under_tol."""
    dev = mirror_deviation(df, p)
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(dev, nan=0.0) < -p["under_tol"]


def _rolling_count(flags: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(flags.astype(float)).rolling(window, min_periods=1).sum().to_numpy()


def score_from(over: np.ndarray, conflict: np.ndarray, p: Mapping[str, Any],
               mirror: np.ndarray | None = None) -> np.ndarray:
    """max of each check's normalised score (>= 1 means alarm)."""
    score = np.maximum(_rolling_count(over, p["window"]) / p["min_violations"],
                       _rolling_count(conflict, p["window"]) / p["cloud_min_violations"])
    if mirror is not None:
        score = np.maximum(score, mirror)
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
    under = m_score = None
    if p["mirror"]:
        dev = mirror_deviation(df, p)
        m_score = mirror_score(dev, p)
        under = np.nan_to_num(dev, nan=0.0) < -p["under_tol"]
    score = score_from(over, conflict, p, m_score)
    alarm = score >= 1.0
    cols = {"score": score, "alarm": alarm, "over_clearsky": over, "cloud_conflict": conflict}
    if under is not None:
        cols["under_clearsky"] = under
    out = pd.DataFrame(cols, index=df.index)

    est_start = first_alarm = None
    hits = np.flatnonzero(alarm)
    if hits.size:
        a = int(hits[0])
        lo = max(0, a - p["window"] + 1)
        bad = over | conflict
        if under is not None:
            bad = bad | under
            clear_pos = np.flatnonzero(~np.isnan(dev[: a + 1]))
            if clear_pos.size:
                lo = min(lo, int(clear_pos[max(0, clear_pos.size - p["under_n"])]))
        hit = np.flatnonzero(bad[lo:a + 1])
        first = lo + int(hit[0]) if hit.size else lo
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


def mirror_natural_alarms(frames: Iterable[pd.DataFrame], params: Mapping[str, Any] | None = None,
                          tol_grid: Sequence[float] = UNDER_TOL_GRID) -> pd.DataFrame:
    """On clean data: mirror-check alarm episodes per location-year for each under_tol.

    Returns a DataFrame indexed by under_tol with columns episodes_per_year, episodes,
    alarm_days, clear_days_share, years.
    """
    frames = list(frames)
    base = merge_params(params)
    devs = [mirror_deviation(df, base) for df in frames]
    years = sum(len(df) for df in frames) / 365.25
    clear_share = (sum(int((~np.isnan(d)).sum()) for d in devs) / max(1, sum(len(df) for df in frames)))
    rows = []
    for tol in tol_grid:
        p = {**base, "under_tol": float(tol)}
        episodes = alarm_days = 0
        for d in devs:
            on = mirror_score(d, p) >= 1.0
            alarm_days += int(on.sum())
            episodes += int(np.sum(on[1:] & ~on[:-1]) + (1 if on.size and on[0] else 0))
        rows.append({"under_tol": float(tol), "episodes_per_year": episodes / years if years else float("nan"),
                     "episodes": episodes, "alarm_days": alarm_days,
                     "clear_days_share": clear_share, "years": years})
    return pd.DataFrame(rows).set_index("under_tol")


def choose_under_tol(frames: Iterable[pd.DataFrame], params: Mapping[str, Any] | None = None,
                     max_episodes_per_year: float = 0.05,
                     tol_grid: Sequence[float] = UNDER_TOL_GRID) -> float:
    """Smallest under_tol whose clean-data alarm episodes per location-year are <= the budget."""
    rates = mirror_natural_alarms(frames, params, tol_grid)
    ok = rates.index[rates["episodes_per_year"] <= max_episodes_per_year]
    return float(ok.min()) if len(ok) else float(max(tol_grid))