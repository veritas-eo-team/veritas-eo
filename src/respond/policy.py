"""Response policy: trust score -> action, with hysteresis.

trust = 1 - p_compromise
  trust >= accept   -> accept
  trust >= deweight -> deweight
  trust >= verify   -> verify
  otherwise         -> quarantine
An exact replay or a stuck run (fused column `forced`) always gives quarantine.

Hysteresis: once in quarantine, the series stays quarantined until it has had
hold_days consecutive days with trust >= accept (released on the hold_days-th such
day); any lower-trust day resets the count. alert is True on the first day of each
quarantine episode.

Sticky release (off unless `sticky` is true; needs a `gap` column): on entering
quarantine the policy records the entry shift, the mean gap over the last
release_window days. Release then also requires the recent mean gap to have fallen
to at most release_frac of the entry shift (same sign), i.e. evidence that the
manipulation has stopped. Entry shifts smaller than min_shift (replay, stuck) keep
the plain rule, and after max_hold_days in quarantine the plain rule applies again.
Causal: only data up to day t is used.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.common import get_logger, load_config

log = get_logger(__name__)

ACTIONS: tuple[str, ...] = ("accept", "deweight", "verify", "quarantine")
DEFAULTS: dict[str, Any] = {
    "accept": 0.8, "deweight": 0.5, "verify": 0.2, "hold_days": 3,
    "sticky": False, "release_window": 14, "release_frac": 0.5,
    "min_shift": 0.01, "max_hold_days": 90,
}


def merge_params(policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """DEFAULTS overridden by non-None values; validated."""
    merged: dict[str, Any] = dict(DEFAULTS)
    for key, value in (policy or {}).items():
        if value is not None:
            merged[key] = value
    for key in ("accept", "deweight", "verify", "release_frac", "min_shift"):
        merged[key] = float(merged[key])
    for key in ("hold_days", "release_window", "max_hold_days"):
        merged[key] = int(merged[key])
    merged["sticky"] = bool(merged["sticky"])
    if not 1.0 >= merged["accept"] > merged["deweight"] > merged["verify"] >= 0.0:
        raise ValueError("policy thresholds must satisfy 1 >= accept > deweight > verify >= 0")
    if merged["hold_days"] < 1:
        raise ValueError("policy.hold_days must be at least 1")
    if merged["release_window"] < 2:
        raise ValueError("policy.release_window must be at least 2")
    if not 0.0 <= merged["release_frac"] < 1.0:
        raise ValueError("policy.release_frac must be in [0, 1)")
    if merged["min_shift"] < 0:
        raise ValueError("policy.min_shift must be >= 0")
    if merged["max_hold_days"] < merged["hold_days"]:
        raise ValueError("policy.max_hold_days must be >= hold_days")
    return merged


def raw_actions(trust: np.ndarray, p: Mapping[str, Any]) -> np.ndarray:
    """Threshold mapping without hysteresis."""
    return np.select(
        [trust >= p["accept"], trust >= p["deweight"], trust >= p["verify"]],
        ["accept", "deweight", "verify"],
        default="quarantine",
    ).astype(object)


def recent_mean_gap(gap: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean of the gap over `window` days (NaN until half the window is valid)."""
    return pd.Series(gap).rolling(window, min_periods=max(2, window // 2)).mean().to_numpy()


def decide(fused: pd.DataFrame, cfg: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """DataFrame[trust, action, alert] on fused.index (causal)."""
    cfg = cfg if cfg is not None else load_config()
    p = merge_params(cfg.get("policy"))
    prob = fused["p_compromise"].astype(float).fillna(0.0).clip(0.0, 1.0).to_numpy()
    trust = 1.0 - prob
    forced = fused["forced"].fillna(False).astype(bool).to_numpy() if "forced" in fused else np.zeros(len(fused), bool)
    raw = raw_actions(trust, p)

    sticky = p["sticky"] and "gap" in fused
    recent = recent_mean_gap(fused["gap"].astype(float).to_numpy(), p["release_window"]) if sticky else None

    n = len(fused)
    action = np.empty(n, dtype=object)
    alert = np.zeros(n, dtype=bool)
    in_quarantine, streak, q_days, entry_shift = False, 0, 0, float("nan")
    for i in range(n):
        if forced[i] or raw[i] == "quarantine":
            if not in_quarantine:
                alert[i] = True
                q_days = 0
                entry_shift = float(recent[i]) if sticky else float("nan")
            in_quarantine, streak = True, 0
            q_days += 1
            action[i] = "quarantine"
        elif in_quarantine:
            q_days += 1
            streak = streak + 1 if trust[i] >= p["accept"] else 0
            release = streak >= p["hold_days"]
            if (release and sticky and q_days < p["max_hold_days"]
                    and entry_shift == entry_shift and abs(entry_shift) >= p["min_shift"]):
                now = recent[i]
                if now == now:  # not NaN
                    release = now * np.sign(entry_shift) <= p["release_frac"] * abs(entry_shift)
            if release:
                in_quarantine, streak = False, 0
                action[i] = raw[i]
            else:
                action[i] = "quarantine"
        else:
            action[i] = raw[i]
    return pd.DataFrame({"trust": trust, "action": action, "alert": alert}, index=fused.index)