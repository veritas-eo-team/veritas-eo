"""Comparison baselines for the benchmark (Phase 2, Step 10).

Each baseline turns a run frame into a contract-C5 result so the benchmark can score it
exactly like VERITAS. All are causal. Thresholds are tuned on the training locations
only (clean years test_year-2 .. test_year-1) to a false-alarm budget, so no method
wins by alarming more often.

  zscore   |7-day mean z| against a causal day-of-year climatology of the site's past years
  cusum    two-sided CUSUM (k = 0.5) on the same climatology z
  iforest  Isolation Forest on log target, target vs neighbours, target vs clear-sky and
           season (fit on training years), 7-day mean anomaly score
  veritas_no_nb  the VERITAS pipeline with the neighbour columns removed (ablation)
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.common import EPS
from src.detect import step
from src.detect.context_model import ClimatologyBaseline, _climatology_log, _log_pos, build_features
from src.pipeline import C5_COLUMNS, run_pipeline

SMOOTH_DAYS = 7
TUNED_KINDS: tuple[str, ...] = ("zscore", "cusum", "iforest")
METHODS: dict[str, str] = {
    "veritas": "VERITAS (full)",
    "veritas_no_nb": "VERITAS without neighbours",
    "cusum": "CUSUM on climatology",
    "zscore": "Z-score on climatology (7-day mean)",
    "iforest": "Isolation Forest (same inputs)",
}


def _smooth(x: np.ndarray, n: int = SMOOTH_DAYS) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=float)).rolling(n, min_periods=1).mean().to_numpy()


def episodes(alarm: np.ndarray) -> int:
    a = np.asarray(alarm, dtype=bool)
    return int(a[0]) + int(np.sum(a[1:] & ~a[:-1])) if a.size else 0


def as_c5(run_df: pd.DataFrame, score: np.ndarray, threshold: float,
          expected: np.ndarray | None = None) -> pd.DataFrame:
    """Baseline score -> C5 result (alarm = quarantine; no attribution, no start estimate)."""
    idx = run_df.index
    n = len(idx)
    s = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0)
    thr = max(float(threshold), 1e-9)
    alarm = s >= thr
    p = 1.0 / (1.0 + np.exp(-np.clip(8.0 * (s / thr - 1.0), -50, 50)))
    target = run_df["target"].to_numpy(dtype=float)
    exp_ = np.asarray(expected, dtype=float) if expected is not None else np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = _log_pos(target) - np.log(np.clip(exp_, 0.0, None) + EPS)
    out = pd.DataFrame({
        "target": target, "expected": exp_, "sigma": np.nan, "gap": gap, "z": np.nan,
        "step_score": np.nan, "drift_score": np.nan, "replay_score": np.nan, "consistency_score": np.nan,
        "p_compromise": p, "trust": 1.0 - p,
        "attack_type": np.where(alarm, "unknown", "none"),
        "est_start": pd.Series(pd.NaT, index=idx, dtype="datetime64[ns]"),
        "action": np.where(alarm, "quarantine", "accept"),
        "alert": alarm & ~np.concatenate([[False], alarm[:-1]]),
    }, index=idx)
    return out[C5_COLUMNS]


class Baselines:
    """Fits the baselines on training frames and provides benchmark pipelines."""

    def __init__(self, cfg: Mapping[str, Any], train_frames: Mapping[str, pd.DataFrame]) -> None:
        self.cfg = cfg
        self.test_year = int(cfg["test_year"])
        self.cv = list(cfg["context_vars"])
        self.clim = ClimatologyBaseline(cfg=cfg).fit(list(train_frames.values()))
        X = []
        for df in train_frames.values():
            f = self._if_features(df[df.index.year < self.test_year])
            X.append(f[~np.isnan(f).any(axis=1)])
        self.iforest = IsolationForest(n_estimators=200, random_state=0).fit(np.vstack(X))
        self.thresholds: dict[str, float] = {}

    # -- features and scores
    def _if_features(self, df: pd.DataFrame) -> np.ndarray:
        F = build_features(df, self.cv)
        lt = _log_pos(df["target"].to_numpy(dtype=float))
        return np.column_stack([lt, lt - F["nb_log_mean"].to_numpy(), lt - F["log_clrsky"].to_numpy(),
                                F["doy_sin"].to_numpy(), F["doy_cos"].to_numpy()])

    def score(self, kind: str, run_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        if kind in ("zscore", "cusum"):
            clim = _climatology_log(run_df, self.clim.window_days)
            z = (_log_pos(run_df["target"].to_numpy(dtype=float)) - clim) / self.clim.sigma_
            expected = np.exp(clim) - EPS
            if kind == "zscore":
                return np.abs(np.nan_to_num(_smooth(z), nan=0.0)), expected
            out = step.detect(pd.Series(z, index=run_df.index), None, {"k": 0.5, "h": 1.0, "phi": 0.0, "cap": 1000.0})
            return out["score"].to_numpy(), expected
        if kind == "iforest":
            F = self._if_features(run_df)
            ok = ~np.isnan(F).any(axis=1)
            raw = np.zeros(len(run_df))
            if ok.any():
                raw[ok] = -self.iforest.score_samples(F[ok])
            return _smooth(raw), None
        raise ValueError(f"unknown baseline: {kind}")

    # -- tuning on training locations only
    def tune(self, kind: str, train_frames: Mapping[str, pd.DataFrame], budget: float) -> tuple[float, float]:
        """Smallest threshold whose clean alarm episodes per location-year on training frames <= budget."""
        ty = self.test_year
        clean = []
        for df in train_frames.values():
            run_df = df.loc[f"{ty - 2}-01-01":f"{ty - 1}-12-31"]
            s, _ = self.score(kind, run_df)
            clean.append(np.nan_to_num(s[run_df.index.year == ty - 1], nan=0.0))
        pooled = np.concatenate(clean)
        years = len(pooled) / 365.25
        grid = np.unique(np.quantile(pooled, np.linspace(0.5, 0.99999, 400)))
        rate = float("nan")
        for thr in grid:
            rate = sum(episodes(x >= thr) for x in clean) / years
            if rate <= budget:
                self.thresholds[kind] = float(thr)
                return float(thr), rate
        self.thresholds[kind] = float(grid[-1])
        return float(grid[-1]), rate

    # -- benchmark pipelines
    def pipeline(self, kind: str) -> Callable[[pd.DataFrame, Any, Mapping[str, Any]], pd.DataFrame]:
        if kind == "veritas":
            return run_pipeline
        if kind == "veritas_no_nb":
            def no_nb(df: pd.DataFrame, model: Any, cfg: Mapping[str, Any]) -> pd.DataFrame:
                return run_pipeline(df.drop(columns=[c for c in df.columns if c.startswith("nb_")]), model, cfg)
            return no_nb
        if kind not in self.thresholds:
            raise RuntimeError(f"tune {kind} before using it")
        thr = self.thresholds[kind]

        def run(df: pd.DataFrame, model: Any, cfg: Mapping[str, Any]) -> pd.DataFrame:
            s, expected = self.score(kind, df)
            return as_c5(df, s, thr, expected)
        return run