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
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.common import get_logger, load_config

log = get_logger(__name__)

ACTIONS: tuple[str, ...] = ("accept", "deweight", "verify", "quarantine")
DEFAULTS: dict[str, float] = {"accept": 0.8, "deweight": 0.5, "verify": 0.2, "hold_days": 3}


def merge_params(policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """DEFAULTS overridden by non-None values; validated."""
    merged: dict[str, Any] = dict(DEFAULTS)
    for key, value in (policy or {}).items():
        if value is not None:
            merged[key] = value
    for key in ("accept", "deweight", "verify"):
        merged[key] = float(merged[key])
    merged["hold_days"] = int(merged["hold_days"])
    if not 1.0 >= merged["accept"] > merged["deweight"] > merged["verify"] >= 0.0:
        raise ValueError("policy thresholds must satisfy 1 >= accept > deweight > verify >= 0")
    if merged["hold_days"] < 1:
        raise ValueError("policy.hold_days must be at least 1")
    return merged


def raw_actions(trust: np.ndarray, p: Mapping[str, Any]) -> np.ndarray:
    """Threshold mapping without hysteresis."""
    return np.select(
        [trust >= p["accept"], trust >= p["deweight"], trust >= p["verify"]],
        ["accept", "deweight", "verify"],
        default="quarantine",
    ).astype(object)


def decide(fused: pd.DataFrame, cfg: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """DataFrame[trust, action, alert] on fused.index (causal)."""
    cfg = cfg if cfg is not None else load_config()
    p = merge_params(cfg.get("policy"))
    prob = fused["p_compromise"].astype(float).fillna(0.0).clip(0.0, 1.0).to_numpy()
    trust = 1.0 - prob
    forced = fused["forced"].fillna(False).astype(bool).to_numpy() if "forced" in fused else np.zeros(len(fused), bool)
    raw = raw_actions(trust, p)

    n = len(fused)
    action = np.empty(n, dtype=object)
    alert = np.zeros(n, dtype=bool)
    in_quarantine, streak = False, 0
    for i in range(n):
        if forced[i] or raw[i] == "quarantine":
            if not in_quarantine:
                alert[i] = True
            in_quarantine, streak = True, 0
            action[i] = "quarantine"
        elif in_quarantine:
            if trust[i] >= p["accept"]:
                streak += 1
                if streak >= p["hold_days"]:
                    in_quarantine, streak = False, 0
                    action[i] = raw[i]
                else:
                    action[i] = "quarantine"
            else:
                streak = 0
                action[i] = "quarantine"
        else:
            action[i] = raw[i]
    return pd.DataFrame({"trust": trust, "action": action, "alert": alert}, index=fused.index)