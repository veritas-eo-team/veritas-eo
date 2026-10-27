"""Fuse the four detector scores into p_compromise and an attack type.

Input `det` (one row per day): step_score, drift_score, replay_score, consistency_score,
replay_exact, stuck, gap, step_est_start, drift_est_start (Timestamp or NaT, filled
from each detector's first alarm onward). Missing score columns count as 0, NaN as 0.

rules mode: p_i = sigmoid(slope * (score_i - 1)) for the four scores;
            p = 1 - prod(1 - p_i). A score of exactly 1 (the alarm threshold) gives p_i = 0.5.
lr mode:    LogisticRegression on the four scores and their causal 7-day maxima,
            loaded from models/fusion.joblib (falls back to rules if the file is missing).
Both modes: an exact replay or a stuck run forces p >= 0.99 (column `forced`).

attack_type (causal, per day): "stuck" / "replay" if those fired; otherwise, if step or
drift alarmed, classify_shape on the gap since the earlier of step_est_start /
drift_est_start; if only consistency alarmed, classify_shape on the last 60 days;
else "none". To stay fast the shape is re-fitted every 7 days of an episode (daily for
the first 14 days) on at most 180 days after the start, and cached.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import PATHS, get_logger, load_config
from src.detect.drift import classify_shape

log = get_logger(__name__)

SCORE_COLUMNS: tuple[str, ...] = ("step_score", "drift_score", "replay_score", "consistency_score")
FEATURE_COLUMNS: list[str] = list(SCORE_COLUMNS) + [f"{c}_max7" for c in SCORE_COLUMNS]
ATTACK_TYPES: tuple[str, ...] = ("none", "step", "drift", "replay", "stuck")
DEFAULT_SLOPE = 8.0
FORCED_P = 0.99
CONSISTENCY_LOOKBACK = 60
RECLASSIFY_EVERY = 7
DAILY_SHAPE_DAYS = 14
SHAPE_LOOKBACK = 90  # the shape fit looks back at least this far from the first alarm day
MAX_SHAPE_DAYS = 180
LABEL_LEVEL = 0.75  # step/drift/consistency scores this close to an alarm already set the type
FUSION_MODEL_NAME = "fusion.joblib"


# ------------------------------------------------------------------ inputs

def _scores(det: pd.DataFrame) -> pd.DataFrame:
    """The four scores as float, NaN or missing -> 0."""
    data = {c: det[c].astype(float) if c in det else pd.Series(0.0, index=det.index) for c in SCORE_COLUMNS}
    return pd.DataFrame(data, index=det.index).fillna(0.0).clip(lower=0.0)


def _flag(det: pd.DataFrame, col: str) -> np.ndarray:
    if col not in det:
        return np.zeros(len(det), dtype=bool)
    return det[col].fillna(False).astype(bool).to_numpy()


def _dates(det: pd.DataFrame, col: str) -> pd.DatetimeIndex:
    if col not in det:
        return pd.DatetimeIndex([pd.NaT] * len(det))
    return pd.DatetimeIndex(pd.to_datetime(det[col]))


def fusion_features(det: pd.DataFrame) -> pd.DataFrame:
    """The four scores and their causal rolling 7-day maxima (FEATURE_COLUMNS order)."""
    scores = _scores(det)
    maxima = scores.rolling(7, min_periods=1).max()
    maxima.columns = [f"{c}_max7" for c in SCORE_COLUMNS]
    return pd.concat([scores, maxima], axis=1)[FEATURE_COLUMNS]


# ------------------------------------------------------------------ probability

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def rules_probability(scores: pd.DataFrame, slope: float = DEFAULT_SLOPE) -> np.ndarray:
    """p = 1 - prod_i (1 - sigmoid(slope * (score_i - 1)))."""
    p_i = _sigmoid(slope * (scores[list(SCORE_COLUMNS)].to_numpy(dtype=float) - 1.0))
    return 1.0 - np.prod(1.0 - p_i, axis=1)


def fit_fusion(X: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series) -> Pipeline:
    """Fit the LR fusion model on fusion_features() rows and 0/1 labels."""
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("fit_fusion needs both clean (0) and attacked (1) rows")
    X = X[FEATURE_COLUMNS].to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    model = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    return model.fit(X, y)


def save_fusion(model: Pipeline, path: Path | str | None = None) -> Path:
    path = Path(path) if path is not None else PATHS["models"] / FUSION_MODEL_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": FEATURE_COLUMNS, "model": model}, path)
    return path


def load_fusion(path: Path | str | None = None) -> Pipeline:
    path = Path(path) if path is not None else PATHS["models"] / FUSION_MODEL_NAME
    if not path.exists():
        raise FileNotFoundError("no fusion model; run: python -m src.eval.tune")
    state = joblib.load(path)
    if state.get("features") != FEATURE_COLUMNS:
        raise ValueError("saved fusion model uses different features; re-run tuning")
    return state["model"]


# ------------------------------------------------------------------ attack type

def attack_types(det: pd.DataFrame, scores: pd.DataFrame | None = None) -> np.ndarray:
    """Causal per-day attack type (see module docstring).

    For step, drift and consistency alarms the shape is fitted from an anchor fixed when the
    alarm episode starts: the earliest detector start estimate, but at least SHAPE_LOOKBACK
    days before that day, so the fit places the change point itself instead of trusting a
    late start estimate (a slow ramp seen only from its middle looks like a step).
    """
    scores = scores if scores is not None else _scores(det)
    index = det.index
    n = len(det)
    labels = np.full(n, "none", dtype=object)
    stuck = _flag(det, "stuck")
    exact = _flag(det, "replay_exact")
    alarm = {c: scores[c].to_numpy() >= (1.0 if c == "replay_score" else LABEL_LEVEL) for c in SCORE_COLUMNS}
    step_est = _dates(det, "step_est_start")
    drift_est = _dates(det, "drift_est_start")
    gap = det["gap"].astype(float) if "gap" in det else None
    cache: dict[tuple[pd.Timestamp, pd.Timestamp], str] = {}
    anchor = None

    for i in range(n):
        if stuck[i]:
            labels[i] = "stuck"
            continue
        if exact[i] or alarm["replay_score"][i]:
            labels[i] = "replay"
            continue
        end = index[i]
        if not (alarm["step_score"][i] or alarm["drift_score"][i] or alarm["consistency_score"][i]):
            anchor = None
            continue
        if anchor is None:
            starts = [d for d in (step_est[i], drift_est[i]) if not pd.isna(d) and d <= end]
            anchor = min(starts + [end - pd.Timedelta(days=SHAPE_LOOKBACK)])
        if gap is None:
            labels[i] = "step"
            continue
        span = (end - anchor).days
        if span > DAILY_SHAPE_DAYS:
            span = min(span - span % RECLASSIFY_EVERY, MAX_SHAPE_DAYS)
        end_eff = anchor + pd.Timedelta(days=span)
        key = (anchor, end_eff)
        if key not in cache:
            cache[key] = classify_shape(gap, anchor, end_eff)
        labels[i] = cache[key]
    return labels


# ------------------------------------------------------------------ fusion

def fuse(det: pd.DataFrame, cfg: Mapping[str, Any] | None = None, lr_model: Pipeline | None = None) -> pd.DataFrame:
    """DataFrame[p_compromise, attack_type, forced] on det.index.

    Uses LR if lr_model is given or cfg fusion.mode is "lr" (loading models/fusion.joblib;
    rules if that file is missing), otherwise the rules formula.
    """
    cfg = cfg if cfg is not None else load_config()
    section = dict(cfg.get("fusion") or {})
    slope = float(section.get("slope") or DEFAULT_SLOPE)
    scores = _scores(det)

    model = lr_model
    if model is None and section.get("mode") == "lr":
        try:
            model = load_fusion()
        except FileNotFoundError:
            log.warning("fusion mode is 'lr' but no fusion model exists; using rules")
    if model is not None:
        X = fusion_features(det).to_numpy(dtype=float)
        p = model.predict_proba(X)[:, 1] if len(X) else np.zeros(0)
    else:
        p = rules_probability(scores, slope)

    forced = _flag(det, "replay_exact") | _flag(det, "stuck")
    p = np.where(forced, np.maximum(p, FORCED_P), p)
    p = np.clip(np.nan_to_num(p, nan=0.0), 0.0, 1.0)
    return pd.DataFrame({"p_compromise": p, "attack_type": attack_types(det, scores), "forced": forced},
                        index=det.index)