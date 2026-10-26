"""Drift detector (contract C4): EWMA on prewhitened z plus a rolling slope t-test on the gap.

Signal 1 (EWMA):  w_t = (1 - lambda) w_{t-1} + lambda e_t, e = AR(1)-prewhitened z
                  s1 = |w| / (L * sqrt(lambda / (2 - lambda)))
Signal 2 (slope): OLS slope of the gap over the last slope_window days; its t-statistic,
                  deflated by sqrt((1 + phi) / (1 - phi)) for autocorrelation;
                  s2 = |t| / t_thr
score = max(s1, s2); alarm = score >= 1. Both signals are causal. On days where z or
gap is NaN the score is carried forward, so a NaN day never starts an alarm.

est_start: a continuous piecewise-linear "flat -> ramp -> hold" fit (fit_ramp) on the
gap over `lookback` days ending at the first alarm. classify_shape() uses the same fit:
a ramp of 1 day is a step, so it compares the best step fit with the best ramp fit.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.common import get_logger, load_config
from src.detect.step import count_episodes, prewhiten  # noqa: F401  (re-exported)

log = get_logger(__name__)

DEFAULTS: dict[str, float] = {
    "lambda": 0.05,
    "L": 3.5,
    "slope_window": 60.0,
    "t_thr": 4.0,
    "phi": 0.0,
    "lookback": 180.0,
    "min_valid_frac": 0.5,
}
DURATIONS: tuple[int, ...] = (1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 365)
MIN_RAMP_DAYS = 5
SHAPE_PRE_DAYS = 30
SHAPE_PENALTY = 0.02
MIN_BEFORE = 5
MIN_AFTER = 3
HINGE_DAYS = 100_000


# ------------------------------------------------------------------ params

def merge_params(params: Mapping[str, Any] | None = None) -> dict[str, float]:
    """DEFAULTS overridden by non-None values in `params`; validated."""
    merged = dict(DEFAULTS)
    for key, value in (params or {}).items():
        if value is not None:
            merged[key] = float(value)
    if not 0.0 < merged["lambda"] <= 1.0:
        raise ValueError("drift.lambda must be in (0, 1]")
    if merged["L"] <= 0 or merged["t_thr"] <= 0:
        raise ValueError("drift.L and drift.t_thr must be positive")
    if merged["slope_window"] < 10:
        raise ValueError("drift.slope_window must be at least 10 days")
    if not -1.0 < merged["phi"] < 1.0:
        raise ValueError("drift.phi must be in (-1, 1)")
    if merged["lookback"] < 20:
        raise ValueError("drift.lookback must be at least 20 days")
    if not 0.0 < merged["min_valid_frac"] <= 1.0:
        raise ValueError("drift.min_valid_frac must be in (0, 1]")
    return merged


def params_from_config(cfg: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Drift parameters from config; phi falls back to step.phi (one AR(1) model for z)."""
    cfg = cfg if cfg is not None else load_config()
    params = dict(cfg.get("drift") or {})
    if params.get("phi") is None:
        params["phi"] = (cfg.get("step") or {}).get("phi")
    return merge_params(params)


# ------------------------------------------------------------------ signals

def ewma(e: np.ndarray, lam: float) -> np.ndarray:
    """EWMA starting at 0; NaN days carry the previous value."""
    out = np.empty(len(e))
    w = 0.0
    for i, x in enumerate(np.asarray(e, dtype=float).tolist()):
        if x == x:
            w = (1.0 - lam) * w + lam * x
        out[i] = w
    return out


def rolling_slope_t(gap: np.ndarray, window: int, min_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling OLS slope (per day) and its t-statistic over the last `window` days.

    NaN days are skipped inside the window; fewer than `min_count` valid days -> NaN.
    """
    g = np.asarray(gap, dtype=float)
    n = len(g)
    valid = ~np.isnan(g)
    x = np.where(valid, np.arange(n, dtype=float), 0.0)
    y = np.where(valid, g, 0.0)
    one = valid.astype(float)

    hi = np.arange(1, n + 1)
    lo = np.maximum(0, hi - int(window))

    def rsum(a: np.ndarray) -> np.ndarray:
        c = np.concatenate([[0.0], np.cumsum(a)])
        return c[hi] - c[lo]

    cnt, sx, sy = rsum(one), rsum(x), rsum(y)
    sxx, sxy, syy = rsum(x * x), rsum(x * y), rsum(y * y)
    with np.errstate(divide="ignore", invalid="ignore"):
        sxx_c = sxx - sx * sx / cnt
        sxy_c = sxy - sx * sy / cnt
        syy_c = syy - sy * sy / cnt
        slope = sxy_c / sxx_c
        ssr = np.maximum(syy_c - slope * sxy_c, 0.0)
        se = np.sqrt(ssr / (cnt - 2.0) / sxx_c)
        t = slope / se
    bad = (cnt < max(int(min_count), 3)) | ~np.isfinite(t)
    slope[bad] = np.nan
    t[bad] = np.nan
    return slope, t


# ------------------------------------------------------------------ shape fitting

def fit_ramp(
    t: Sequence[float],
    y: Sequence[float],
    durations: Sequence[int] = DURATIONS,
    min_ramp_days: int = MIN_RAMP_DAYS,
    min_before: int = MIN_BEFORE,
    min_after: int = MIN_AFTER,
) -> dict[str, float] | None:
    """Least-squares fit of y = a + c * r(t), r = clip((t - tau + 1) / D, 0, 1).

    D = 1 is a step at tau; D > 1 is a ramp starting at tau that holds after D days
    (the same shape as inject's drift). Searches tau over observed days (with
    min_before points before and min_after from tau on) and D over `durations`.
    Returns step_*, drift_* (D >= min_ramp_days) and best_* (sse, tau, D, amplitude),
    or None if there are too few points.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < min_before + min_after + 2:
        return None
    taus = t[min_before: n - min_after + 1]
    if taus.size == 0:
        return None
    yc = y - y.mean()
    syy = float(yc @ yc)
    results: list[tuple[float, float, int, float]] = []
    for d in durations:
        r = np.clip((t[None, :] - taus[:, None] + 1.0) / float(d), 0.0, 1.0)
        rc = r - r.mean(axis=1, keepdims=True)
        srr = np.einsum("ij,ij->i", rc, rc)
        sry = rc @ yc
        with np.errstate(divide="ignore", invalid="ignore"):
            sse = np.where(srr > 1e-12, syy - sry ** 2 / srr, np.inf)
            amp = np.where(srr > 1e-12, sry / srr, 0.0)
        j = int(np.argmin(sse))
        results.append((float(sse[j]), float(taus[j]), int(d), float(amp[j])))

    def pack(prefix: str, item: tuple[float, float, int, float]) -> dict[str, float]:
        return {f"{prefix}_sse": item[0], f"{prefix}_tau": item[1],
                f"{prefix}_D": item[2], f"{prefix}_amp": item[3]}

    out: dict[str, float] = {}
    steps = [r for r in results if r[2] == 1]
    drifts = [r for r in results if r[2] >= min_ramp_days]
    if steps:
        out.update(pack("step", min(steps)))
    if drifts:
        out.update(pack("drift", min(drifts)))
    out.update(pack("best", min(results)))
    return out


def classify_shape(
    gap: pd.Series,
    start: Any,
    end: Any,
    pre_days: int = SHAPE_PRE_DAYS,
    min_ramp_days: int = MIN_RAMP_DAYS,
    penalty: float = SHAPE_PENALTY,
) -> str:
    """"step" or "drift" for the gap from (start - pre_days) to end.

    "drift" if the best ramp (>= min_ramp_days) has an SSE more than `penalty` below
    the best step; otherwise "step" (also when there is too little data).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window = gap.loc[start - pd.Timedelta(days=pre_days): end].dropna()
    if len(window) < MIN_BEFORE + MIN_AFTER + 5:
        return "step"
    t = (window.index - window.index[0]).days.to_numpy(dtype=float)
    fit = fit_ramp(t, window.to_numpy(dtype=float), min_ramp_days=min_ramp_days)
    if fit is None or "drift_sse" not in fit or "step_sse" not in fit:
        return "step"
    return "drift" if fit["drift_sse"] < (1.0 - penalty) * fit["step_sse"] else "step"


# ------------------------------------------------------------------ detector

def detect(z: pd.Series, gap: pd.Series | None = None, params: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Contract C4: DataFrame[score, alarm] on z.index.

    attrs: est_start (Timestamp or None), first_alarm, direction ("up"/"down"/None),
    trigger ("ewma"/"slope"/None), ramp_days (D of the fitted ramp, 1 = step-like).
    Without `gap`, only the EWMA signal is used and est_start is fitted on z.
    """
    p = merge_params(params)
    lam, phi = p["lambda"], p["phi"]
    index = z.index
    zv = z.to_numpy(dtype=float)
    gv = gap.reindex(index).to_numpy(dtype=float) if gap is not None else None

    w = ewma(prewhiten(zv, phi), lam)
    s1 = np.abs(w) / (p["L"] * math.sqrt(lam / (2.0 - lam)))
    if gv is not None:
        window = int(p["slope_window"])
        slope, t = rolling_slope_t(gv, window, max(10, int(p["min_valid_frac"] * window)))
        t_adj = t / math.sqrt((1.0 + phi) / (1.0 - phi))
        s2 = np.nan_to_num(np.abs(t_adj) / p["t_thr"], nan=0.0)
    else:
        slope = np.full(len(zv), np.nan)
        s2 = np.zeros(len(zv))

    invalid = np.isnan(zv) | (np.isnan(gv) if gv is not None else False)
    raw = np.maximum(s1, s2)
    score = pd.Series(np.where(invalid, np.nan, raw)).ffill().fillna(0.0).to_numpy()
    alarm = score >= 1.0
    out = pd.DataFrame({"score": score, "alarm": alarm}, index=index)

    est_start = first_alarm = direction = trigger = None
    ramp_days = None
    hits = np.flatnonzero(alarm)
    if hits.size:
        a = int(hits[0])
        first_alarm = index[a]
        by_ewma = s1[a] >= s2[a]
        trigger = "ewma" if by_ewma else "slope"
        sign = w[a] if by_ewma else slope[a]
        direction = "up" if (sign == sign and sign >= 0) else "down"
        lo = max(0, a - int(p["lookback"]) + 1)
        series = gv if gv is not None else zv
        seg = series[lo: a + 1]
        pos = np.arange(lo, a + 1, dtype=float)
        ok = ~np.isnan(seg)
        hinge = fit_ramp(pos[ok], seg[ok], durations=(HINGE_DAYS,)) if ok.sum() else None
        shape = fit_ramp(pos[ok], seg[ok]) if ok.sum() else None
        est_start = index[int(hinge["best_tau"])] if hinge is not None else index[lo]
        ramp_days = int(shape["best_D"]) if shape is not None else None
    out.attrs.update(est_start=est_start, first_alarm=first_alarm, direction=direction,
                     trigger=trigger, ramp_days=ramp_days)
    return out