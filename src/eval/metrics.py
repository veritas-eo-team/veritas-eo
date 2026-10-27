"""Evaluation metrics for the benchmark.

"alarm" = action in {verify, quarantine}. All metrics use the test-year rows only.
event:       caught (alarm within [start, end + 30 days]), delay_days (first such alarm -
             start), false-alarm episodes (rising edges of the alarm on clean days: not
             attacked and not in the 30-day grace period after the attack), days observed
per-day:     precision, recall, F1 of the alarm and ROC-AUC of p_compromise against the
             attack mask (grace days excluded)
calibration: Brier score and a 10-bin reliability table of p_compromise
attribution: attack_type at the first alarm vs the injected type; est_start error in days
per-detector: which detector's score reached 1 first between the start and the first alarm
cost:        ms per location-day and peak memory are measured in run_benchmark;
             system_info() describes the laptop (no host or user names)
"""
from __future__ import annotations

import platform
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ALARM_ACTIONS: tuple[str, ...] = ("verify", "quarantine")
GRACE_DAYS = 30
SETTLE_DAYS = 30  # attack type is also scored this many days after the first alarm
DAYS_PER_YEAR = 365.25
DETECTOR_ORDER: tuple[str, ...] = ("replay", "step", "drift", "consistency")
EXPECTED_TYPE: dict[str, str] = {"step": "step", "drift": "drift", "replay": "replay", "stuck": "stuck"}


def alarm_mask(result: pd.DataFrame) -> np.ndarray:
    return result["action"].isin(ALARM_ACTIONS).to_numpy()


def rising_edges(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=bool)
    return a & ~np.concatenate([[False], a[:-1]])


def attack_windows(index: pd.DatetimeIndex, labels: Mapping[str, Any],
                   grace_days: int = GRACE_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """(attacked, grace) boolean arrays on `index`."""
    if labels.get("type", "none") == "none" or labels.get("start") is None:
        empty = np.zeros(len(index), dtype=bool)
        return empty, empty.copy()
    start, end = pd.Timestamp(labels["start"]), pd.Timestamp(labels["end"])
    attacked = np.asarray((index >= start) & (index <= end))
    grace = np.asarray((index > end) & (index <= end + pd.Timedelta(days=grace_days)))
    return attacked, grace


def fa_per_location_year(episodes: float, days: float) -> float:
    return float(episodes) / (float(days) / DAYS_PER_YEAR) if days > 0 else float("nan")


def event_metrics(result: pd.DataFrame, labels: Mapping[str, Any], test_year: int | None = None,
                  grace_days: int = GRACE_DAYS) -> dict[str, Any]:
    """Event-level metrics for one run (C5 result + inject labels)."""
    r = result if test_year is None else result.loc[result.index.year == int(test_year)]
    index = r.index
    alarm = alarm_mask(r)
    attacked, grace = attack_windows(index, labels, grace_days)
    clean = ~attacked & ~grace
    rec: dict[str, Any] = {
        "n_days": int(len(r)),
        "fa_episodes": int(np.sum(rising_edges(alarm) & clean)),
        "fa_days": int(np.sum(clean)),
        "caught": np.nan, "delay_days": np.nan, "type_at_alarm": None, "type_correct": np.nan,
        "start_error_days": np.nan, "first_detector": None, "min_trust_attacked": np.nan,
        "type_30d": None, "type_30d_correct": np.nan,
    }
    if labels.get("type", "none") == "none" or labels.get("start") is None:
        return rec
    start, end = pd.Timestamp(labels["start"]), pd.Timestamp(labels["end"])
    window = np.asarray((index >= start) & (index <= end + pd.Timedelta(days=grace_days)))
    hits = np.flatnonzero(alarm & window)
    rec["caught"] = bool(hits.size)
    if attacked.any():
        rec["min_trust_attacked"] = float(r["trust"].to_numpy()[attacked].min())
    if not hits.size:
        return rec
    a = int(hits[0])
    first = index[a]
    rec["delay_days"] = int((first - start).days)
    rec["type_at_alarm"] = str(r["attack_type"].iloc[a])
    expected = EXPECTED_TYPE.get(str(labels["type"]))
    rec["type_correct"] = (rec["type_at_alarm"] == expected) if expected else np.nan
    settle = window & alarm & np.asarray(index <= first + pd.Timedelta(days=SETTLE_DAYS))
    seen = [str(t) for t in r["attack_type"].to_numpy()[settle] if str(t) != "none"]
    if "replay" in seen:
        later = "replay"
    elif "stuck" in seen:
        later = "stuck"
    else:
        later = seen[-1] if seen else rec["type_at_alarm"]
    rec["type_30d"] = later
    rec["type_30d_correct"] = (later == expected) if expected else np.nan
    est = r["est_start"].iloc[a]
    if not pd.isna(est):
        rec["start_error_days"] = int((pd.Timestamp(est) - start).days)
    between = np.asarray((index >= start) & (index <= first))
    best: tuple[int, int, str] | None = None
    for priority, det in enumerate(DETECTOR_ORDER):
        col = f"{det}_score"
        if col not in r:
            continue
        days = np.flatnonzero((r[col].to_numpy() >= 1.0) & between)
        if days.size and (best is None or (int(days[0]), priority) < best[:2]):
            best = (int(days[0]), priority, det)
    rec["first_detector"] = best[2] if best else "fusion"
    return rec


def per_day_metrics(y: Any, score: Any, alarm: Any) -> dict[str, float]:
    """Precision, recall, F1 of `alarm` and ROC-AUC of `score` against `y`."""
    y = np.asarray(y, dtype=bool)
    alarm = np.asarray(alarm, dtype=bool)
    score = np.asarray(score, dtype=float)
    tp = int(np.sum(alarm & y))
    fp = int(np.sum(alarm & ~y))
    fn = int(np.sum(~alarm & y))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and precision + recall > 0 else float("nan"))
    auc = float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "roc_auc": auc}


def calibration(y: Any, p: Any, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """(Brier score, reliability table [bin_lo, bin_hi, n, mean_p, frac_positive])."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    brier = float(np.mean((p - y) ** 2)) if len(y) else float("nan")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        rows.append({"bin_lo": b / n_bins, "bin_hi": (b + 1) / n_bins, "n": int(m.sum()),
                     "mean_p": float(p[m].mean()) if m.any() else float("nan"),
                     "frac_positive": float(y[m].mean()) if m.any() else float("nan")})
    return brier, pd.DataFrame(rows)


def summarize(records: pd.DataFrame, perday: pd.DataFrame) -> pd.DataFrame:
    """Benchmark table: one row per (group, attack) for the standard and adaptive families."""
    main = records[records["family"].isin(["standard", "adaptive"])].copy()
    main["row"] = np.where(main["family"] == "adaptive", "adaptive_step", main["attack"])
    rows = []
    for (group, row), recs in main.groupby(["group", "row"], sort=False):
        days = perday[perday["scenario_id"].isin(recs["id"]) & ~perday["grace"]]
        attacked = recs[recs["attack"] != "none"]
        caught = attacked[attacked["caught"] == True]  # noqa: E712
        pdm = per_day_metrics(days["y"], days["p"], days["alarm"]) if len(days) else {}
        rows.append({
            "group": group, "attack": row, "n_scenarios": int(len(recs)),
            "catch_rate": float(attacked["caught"].astype(float).mean()) if len(attacked) else np.nan,
            "median_delay_days": float(caught["delay_days"].median()) if len(caught) else np.nan,
            "attribution_acc": float(caught["type_correct"].astype(float).mean()) if len(caught) else np.nan,
            "attribution_acc_30d": float(caught["type_30d_correct"].astype(float).mean()) if len(caught) else np.nan,
            "median_abs_start_error_days": float(caught["start_error_days"].abs().median()) if len(caught) else np.nan,
            "fa_per_location_year": fa_per_location_year(recs["fa_episodes"].sum(), recs["fa_days"].sum()),
            "precision": pdm.get("precision", np.nan), "recall": pdm.get("recall", np.nan),
            "f1": pdm.get("f1", np.nan), "roc_auc": pdm.get("roc_auc", np.nan),
            "brier": calibration(days["y"], days["p"])[0] if len(days) else np.nan,
        })
    return pd.DataFrame(rows)


def measured_limit(records: pd.DataFrame, min_catch: float = 0.5) -> float | None:
    """Smallest positive swept step size caught in >= min_catch of runs (None if none)."""
    sweep = records[(records["family"] == "sweep_step") & (records["size"] > 0)]
    if sweep.empty:
        return None
    rates = sweep.groupby("size")["caught"].apply(lambda s: s.astype(float).mean())
    ok = rates[rates >= min_catch]
    return float(ok.index.min()) if len(ok) else None


def system_info() -> dict[str, Any]:
    """CPU, cores, RAM, OS and Python versions (no host or user names)."""
    import psutil

    return {
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }