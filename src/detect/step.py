"""Step detector (contract C4): two-sided CUSUM on AR(1)-prewhitened z.

    e_t = (z_t - phi * z_prev) / sqrt(1 - phi^2)     z_prev = last valid z before t
    S+_t = min(max(0, S+_{t-1} + e_t - k), cap*h)
    S-_t = min(max(0, S-_{t-1} - e_t - k), cap*h)
    score = max(S+, S-) / h        alarm = score >= 1

The sqrt(1 - phi^2) factor gives e unit variance when z is AR(1) with unit
variance, so k and h are in standard-deviation units. NaN days carry both
statistics forward unchanged, so a NaN day can continue an alarm but never
start one. The cap lets the score fall back soon after an attack ends.

est_start = the day after the last zero of the winning CUSUM before the first
alarm. refine_start() re-estimates the start and size from the gap with ruptures
(used only after an alarm; it looks back, which is fine for attribution).
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import ruptures as rpt

from src.common import get_logger, load_config

log = get_logger(__name__)

DEFAULTS: dict[str, float] = {"k": 0.5, "h": 8.0, "phi": 0.0, "cap": 2.0}
REFINE_LOOKBACK = 90
MIN_SEGMENT = 5


# ------------------------------------------------------------------ params

def merge_params(params: Mapping[str, Any] | None = None) -> dict[str, float]:
    """DEFAULTS overridden by non-None values in `params`; validated."""
    merged = dict(DEFAULTS)
    for key, value in (params or {}).items():
        if value is not None:
            merged[key] = float(value)
    if merged["h"] <= 0:
        raise ValueError("step.h must be positive")
    if merged["k"] < 0:
        raise ValueError("step.k must be >= 0")
    if not -1.0 < merged["phi"] < 1.0:
        raise ValueError("step.phi must be in (-1, 1)")
    if merged["cap"] < 1.0:
        raise ValueError("step.cap must be >= 1")
    return merged


def params_from_config(cfg: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Step parameters from the config `step` section (tuned.yaml included)."""
    cfg = cfg if cfg is not None else load_config()
    return merge_params(cfg.get("step"))


# ------------------------------------------------------------------ helpers

def prewhiten(z: np.ndarray, phi: float) -> np.ndarray:
    """AR(1) prewhitening against the last valid previous value; NaN stays NaN."""
    z = np.asarray(z, dtype=float)
    prev = pd.Series(z).ffill().shift(1).to_numpy()
    e = np.where(np.isnan(prev), z, (z - phi * prev) / math.sqrt(1.0 - phi * phi))
    e[np.isnan(z)] = np.nan
    return e


def count_episodes(alarm: Iterable[bool]) -> int:
    """Number of alarm episodes (rising edges of the alarm)."""
    a = np.asarray(list(alarm), dtype=bool)
    if a.size == 0:
        return 0
    return int(a[0]) + int(np.sum(a[1:] & ~a[:-1]))


def estimate_phi(z_list: Iterable[Any]) -> float:
    """Pooled lag-1 autocorrelation of clean z series (each demeaned), clipped to [-0.9, 0.95]."""
    num = den = 0.0
    for z in z_list:
        x = np.asarray(z, dtype=float)
        if np.all(np.isnan(x)):
            continue
        x = x - np.nanmean(x)
        a, b = x[1:], x[:-1]
        ok = ~np.isnan(a) & ~np.isnan(b)
        num += float(a[ok] @ b[ok])
        den += float(b[ok] @ b[ok])
    if den == 0.0:
        return 0.0
    return float(np.clip(num / den, -0.9, 0.95))


def refine_start(gap: pd.Series, alarm_day: Any, lookback: int = REFINE_LOOKBACK) -> tuple[pd.Timestamp | None, float]:
    """Binary segmentation (l2, 1 breakpoint) of the gap over `lookback` days ending at alarm_day.

    Returns (start, size_pct) with size_pct = exp(mean after - mean before) - 1,
    or (None, nan) if the window is too short.
    """
    alarm_day = pd.Timestamp(alarm_day)
    window = gap.loc[alarm_day - pd.Timedelta(days=lookback): alarm_day].dropna()
    if len(window) < 2 * MIN_SEGMENT:
        return None, float("nan")
    values = window.to_numpy(dtype=float).reshape(-1, 1)
    breakpoint_ = rpt.Binseg(model="l2", min_size=MIN_SEGMENT, jump=1).fit(values).predict(n_bkps=1)[0]
    before, after = values[:breakpoint_, 0], values[breakpoint_:, 0]
    size_pct = float(np.exp(after.mean() - before.mean()) - 1.0)
    return window.index[breakpoint_], size_pct


# ------------------------------------------------------------------ detector

def detect(z: pd.Series, gap: pd.Series | None = None, params: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Contract C4: DataFrame[score, alarm] on z.index.

    attrs: est_start (Timestamp or None), first_alarm, direction ("up"/"down"/None),
    refined_start and size_pct (from refine_start on the gap, if given).
    """
    p = merge_params(params)
    k, h, cap = p["k"], p["h"], p["cap"] * p["h"]
    index = z.index
    e = prewhiten(z.to_numpy(dtype=float), p["phi"])

    n = len(e)
    up = np.empty(n)
    down = np.empty(n)
    s_up = s_down = 0.0
    for i, x in enumerate(e.tolist()):
        if x == x:  # not NaN
            s_up = min(max(0.0, s_up + x - k), cap)
            s_down = min(max(0.0, s_down - x - k), cap)
        up[i] = s_up
        down[i] = s_down

    score = np.maximum(up, down) / h
    alarm = score >= 1.0
    out = pd.DataFrame({"score": score, "alarm": alarm}, index=index)

    est_start = first_alarm = refined = None
    direction = None
    size_pct = float("nan")
    hits = np.flatnonzero(alarm)
    if hits.size:
        a = int(hits[0])
        winner = up if up[a] >= down[a] else down
        direction = "up" if winner is up else "down"
        zeros = np.flatnonzero(winner[: a + 1] == 0.0)
        start_pos = min(int(zeros[-1]) + 1, a) if zeros.size else 0
        est_start = index[start_pos]
        first_alarm = index[a]
        if gap is not None:
            refined, size_pct = refine_start(gap.loc[: index[a]], index[a])
    out.attrs.update(est_start=est_start, first_alarm=first_alarm, direction=direction,
                     refined_start=refined, size_pct=size_pct)
    return out