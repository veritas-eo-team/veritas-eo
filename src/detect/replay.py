"""Replay and stuck-value detector (contract C4).

Exact repeats: the m-day window of the target (rounded to `decimals`) ending at day t
is "exact" if it equals an earlier, non-overlapping window (ending at or before t - m).
Windows are kept in a dict keyed by their rounded values. Near-constant windows
(std < min_window_std, e.g. 7 days of zero rain) are skipped because they repeat
naturally.

Near copies: Euclidean distance (not z-normalised) from the window ending at t to its
nearest earlier non-overlapping window (a left matrix profile, computed directly with
chunked matrix products). near = 1 - d / (sqrt(m) * local_std), where local_std is the
causal rolling std of the target over std_window days. In deserts clear days look alike,
so a near copy only alarms if the context disagrees: rolling mean |z| over the window
> disagree_thr.

Stuck: a run of >= stuck_run identical (rounded) values.

score = max(2.0 if exact, near score, 2.0 if stuck); alarm = score >= 1.
near score = near / near_thr * min(disagreement / disagree_thr, 1), capped at 0.99 when
the context does not disagree. All signals are causal; NaN days score 0.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.common import get_logger, load_config

log = get_logger(__name__)

DEFAULTS: dict[str, Any] = {
    "m": 7,
    "decimals": 2,
    "near_thr": 0.8,
    "disagree_thr": 2.0,
    "stuck_run": 5,
    "min_window_std": 0.02,
    "std_window": 30,
}
INT_KEYS = ("m", "decimals", "stuck_run", "std_window")
EXACT_SCORE = 2.0
STUCK_SCORE = 2.0
NO_DISAGREE_CAP = 0.99
CHUNK = 256


# ------------------------------------------------------------------ params

def merge_params(params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """DEFAULTS overridden by non-None values in `params`; validated."""
    merged = dict(DEFAULTS)
    for key, value in (params or {}).items():
        if value is not None:
            merged[key] = value
    for key in INT_KEYS:
        merged[key] = int(merged[key])
    for key in ("near_thr", "disagree_thr", "min_window_std"):
        merged[key] = float(merged[key])
    if merged["m"] < 2:
        raise ValueError("replay.m must be at least 2")
    if merged["stuck_run"] < 2:
        raise ValueError("replay.stuck_run must be at least 2")
    if not 0.0 < merged["near_thr"] <= 1.0:
        raise ValueError("replay.near_thr must be in (0, 1]")
    if merged["disagree_thr"] <= 0:
        raise ValueError("replay.disagree_thr must be positive")
    if merged["std_window"] < 3:
        raise ValueError("replay.std_window must be at least 3")
    return merged


def params_from_config(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Replay parameters from the config `replay` section (tuned.yaml included)."""
    cfg = cfg if cfg is not None else load_config()
    return merge_params(cfg.get("replay"))


# ------------------------------------------------------------------ building blocks

def window_stats(x: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(windows, valid, std) for every m-day window; window k covers days k..k+m-1."""
    if len(x) < m:
        return np.empty((0, m)), np.zeros(0, dtype=bool), np.zeros(0)
    w = sliding_window_view(np.asarray(x, dtype=float), m)
    valid = ~np.isnan(w).any(axis=1)
    std = np.full(len(w), np.nan)
    if valid.any():
        std[valid] = w[valid].std(axis=1)
    return w, valid, std


def exact_matches(x: np.ndarray, m: int, decimals: int, eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per day: True if the window ending there equals an earlier non-overlapping window.

    Returns (exact[n], source window start position[n], -1 if none). The earliest
    matching window is reported as the source.
    """
    n = len(x)
    exact = np.zeros(n, dtype=bool)
    source = np.full(n, -1, dtype=int)
    n_windows = n - m + 1
    if n_windows <= 0:
        return exact, source
    q = np.rint(np.where(np.isnan(x), 0.0, x) * 10.0 ** decimals).astype(np.int64)
    seen: dict[bytes, int] = {}
    for kt in range(n_windows):
        kc = kt - m  # newest window that no longer overlaps the query window
        if kc >= 0 and eligible[kc]:
            seen.setdefault(q[kc:kc + m].tobytes(), kc)
        if eligible[kt]:
            j = seen.get(q[kt:kt + m].tobytes())
            if j is not None:
                exact[kt + m - 1] = True
                source[kt + m - 1] = j
    return exact, source


def nearest_earlier(windows: np.ndarray, eligible: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Euclidean distance from each window to its nearest earlier window starting <= k - m.

    Returns (distance[n_windows], source start[n_windows]); inf / -1 if none.
    """
    n_windows = len(windows)
    dist = np.full(n_windows, np.inf)
    source = np.full(n_windows, -1, dtype=int)
    if n_windows <= m:
        return dist, source
    wf = np.where(eligible[:, None], windows, 0.0)
    norms = np.einsum("ij,ij->i", wf, wf)
    for c0 in range(m, n_windows, CHUNK):
        rows = np.arange(c0, min(n_windows, c0 + CHUNK))
        width = int(rows[-1]) - m + 1
        d2 = norms[rows, None] + norms[None, :width] - 2.0 * (wf[rows] @ wf[:width].T)
        cols = np.arange(width)
        bad = (cols[None, :] > rows[:, None] - m) | ~eligible[None, :width] | ~eligible[rows][:, None]
        d2[bad] = np.inf
        j = np.argmin(d2, axis=1)
        best = d2[np.arange(len(rows)), j]
        ok = np.isfinite(best)
        dist[rows[ok]] = np.sqrt(np.maximum(best[ok], 0.0))
        source[rows[ok]] = j[ok]
    return dist, source


def stuck_runs(x: np.ndarray, decimals: int, run: int) -> tuple[np.ndarray, np.ndarray]:
    """(stuck[n], run start position[n]): True once >= `run` identical rounded values in a row."""
    q = np.rint(np.asarray(x, dtype=float) * 10.0 ** decimals)
    n = len(q)
    length = np.zeros(n, dtype=int)
    start = np.arange(n)
    count, prev, s = 0, np.nan, 0
    for i, v in enumerate(q.tolist()):
        if v != v:
            count, prev = 0, np.nan
            continue
        if count > 0 and v == prev:
            count += 1
        else:
            count, s = 1, i
        prev = v
        length[i] = count
        start[i] = s
    return length >= run, start


def rolling_disagreement(z: np.ndarray, m: int) -> np.ndarray:
    """Causal rolling mean of |z| over m days (needs half the window); NaN -> 0."""
    values = pd.Series(np.abs(np.asarray(z, dtype=float)))
    return values.rolling(m, min_periods=math.ceil(m / 2)).mean().fillna(0.0).to_numpy()


# ------------------------------------------------------------------ detector

def detect(target: pd.Series, z: pd.Series | None = None, params: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Contract C4: DataFrame[score, alarm, exact, stuck] on target.index.

    attrs: est_start (first day of the matched window or stuck run at the first alarm),
    source_start (start of the copied window, None for stuck), match ("exact"/"near"/
    "stuck"/None), first_alarm.
    """
    p = merge_params(params)
    m = p["m"]
    index = target.index
    x = target.to_numpy(dtype=float)
    n = len(x)

    windows, valid, wstd = window_stats(x, m)
    eligible = valid & (np.nan_to_num(wstd, nan=0.0) >= p["min_window_std"])
    exact, exact_src = exact_matches(x, m, p["decimals"], eligible)
    dist, near_src_w = nearest_earlier(windows, eligible, m)

    local_std = target.rolling(p["std_window"], min_periods=10).std().to_numpy()
    near = np.zeros(n)
    near_src = np.full(n, -1, dtype=int)
    if len(windows):
        days = np.arange(m - 1, n)
        k = days - m + 1
        ls = local_std[days]
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.clip(1.0 - dist[k] / (math.sqrt(m) * ls), 0.0, 1.0)
        ok = eligible[k] & np.isfinite(dist[k]) & (ls > 0)
        near[days[ok]] = values[ok]
        near_src[days[ok]] = near_src_w[k][ok]

    if z is not None:
        disagree = rolling_disagreement(z.reindex(index).to_numpy(dtype=float), m)
    else:
        disagree = np.zeros(n)
    s_near = near / p["near_thr"] * np.minimum(disagree / p["disagree_thr"], 1.0)
    s_near = np.where(disagree > p["disagree_thr"], s_near, np.minimum(s_near, NO_DISAGREE_CAP))

    stuck, run_start = stuck_runs(x, p["decimals"], p["stuck_run"])
    score = np.maximum.reduce([np.where(exact, EXACT_SCORE, 0.0), s_near, np.where(stuck, STUCK_SCORE, 0.0)])
    score[np.isnan(x)] = 0.0
    alarm = score >= 1.0
    out = pd.DataFrame({"score": score, "alarm": alarm, "exact": exact, "stuck": stuck}, index=index)

    est_start = source_start = match = first_alarm = None
    hits = np.flatnonzero(alarm)
    if hits.size:
        a = int(hits[0])
        first_alarm = index[a]
        if exact[a]:
            match, est_start, source_start = "exact", index[a - m + 1], index[exact_src[a]]
        elif stuck[a]:
            match, est_start = "stuck", index[run_start[a]]
        else:
            match, est_start = "near", index[a - m + 1]
            source_start = index[near_src[a]] if near_src[a] >= 0 else None
    out.attrs.update(est_start=est_start, source_start=source_start, match=match, first_alarm=first_alarm)
    return out