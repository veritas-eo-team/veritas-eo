"""Expected-value context model (contract C3).

Predicts each day's target from context the attacker did not touch (neighbour
cells, other NASA variables at the site, season, Sentinel-2), never from the
site's own target. Then

    gap = log(target + EPS) - log(expected + EPS)      z = gap / sigma

A 12% step in the target is a constant +0.113 shift in the gap.

Mean model: Pipeline(median imputer, StandardScaler, Ridge) on log(target + EPS).
Uncertainty: out-of-fold residuals (GroupKFold by location); a second Ridge
predicts log|residual| from the same features; sigma = exp(pred) * c, with c set
so the out-of-fold median |z| is 0.674 (as for a standard normal); floor sigma_min.

Training uses only rows with a target from years before test_year, and refuses
frames whose attrs split is "test".

CLI: python -m src.detect.context_model --train
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import EPS, PATHS, get_logger, load_config

log = get_logger(__name__)

MODEL_VERSION = 1
CLRSKY_VAR = "CLRSKY_SFC_SW_DWN"
LOG1P_VARS = frozenset({"PRECTOTCORR"})  # skewed, non-negative
DSC_CAP = 60.0
MEDIAN_ABS_Z = 0.6744897501960817       # median |N(0, 1)|
RESIDUAL_FLOOR = 1e-4                   # keeps log|residual| finite
DEFAULTS: dict[str, float] = {"alpha": 1.0, "sigma_min": 0.02}
OUTPUT_COLUMNS: list[str] = ["expected", "sigma", "gap", "z"]
DEFAULT_MODEL_NAME = "context_model.joblib"


# ------------------------------------------------------------------ features

def _log_pos(x: np.ndarray) -> np.ndarray:
    """log(max(x, 0) + EPS); NaN stays NaN."""
    return np.log(np.clip(np.asarray(x, dtype=float), 0.0, None) + EPS)


def feature_names(context_vars: Sequence[str]) -> list[str]:
    """Fixed feature list (values, then one missing-indicator per value)."""
    others = [f"ctx_{var}" for var in context_vars if var != CLRSKY_VAR]
    values = (["nb_log_mean", "nb_log_median", "nb_log_std", "n_neighbors", "log_clrsky"]
              + others + ["doy_sin", "doy_cos", "ndvi", "days_since_clear_capped"])
    return values + [f"{name}_missing" for name in values]


def build_features(df: pd.DataFrame, context_vars: Sequence[str]) -> pd.DataFrame:
    """Row-wise features; never uses the target column. Same width for any neighbour count."""
    n = len(df)
    nan = np.full(n, np.nan)
    feats: dict[str, np.ndarray] = {}

    nb_cols = sorted(c for c in df.columns if c.startswith("nb_"))
    if nb_cols:
        nb = _log_pos(df[nb_cols].to_numpy(dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
            feats["nb_log_mean"] = np.nanmean(nb, axis=1)
            feats["nb_log_median"] = np.nanmedian(nb, axis=1)
            feats["nb_log_std"] = np.nanstd(nb, axis=1)
        feats["n_neighbors"] = np.sum(~np.isnan(nb), axis=1).astype(float)
    else:
        feats["nb_log_mean"] = nan.copy()
        feats["nb_log_median"] = nan.copy()
        feats["nb_log_std"] = nan.copy()
        feats["n_neighbors"] = np.zeros(n)

    clr_col = f"ctx_{CLRSKY_VAR}"
    feats["log_clrsky"] = _log_pos(df[clr_col].to_numpy(dtype=float)) if clr_col in df else nan.copy()

    for var in context_vars:
        if var == CLRSKY_VAR:
            continue
        col = f"ctx_{var}"
        values = df[col].to_numpy(dtype=float) if col in df else nan.copy()
        feats[col] = np.log1p(np.clip(values, 0.0, None)) if var in LOG1P_VARS else values

    doy = df.index.dayofyear.to_numpy(dtype=float)
    angle = 2.0 * np.pi * (doy - 1.0) / 365.25
    feats["doy_sin"] = np.sin(angle)
    feats["doy_cos"] = np.cos(angle)
    feats["ndvi"] = df["ndvi"].to_numpy(dtype=float) if "ndvi" in df else nan.copy()
    dsc = df["days_since_clear"].to_numpy(dtype=float) if "days_since_clear" in df else nan.copy()
    feats["days_since_clear_capped"] = np.minimum(dsc, DSC_CAP)

    for name in list(feats):
        feats[f"{name}_missing"] = np.isnan(feats[name]).astype(float)
    out = pd.DataFrame(feats, index=df.index)
    return out[feature_names(context_vars)]


# ------------------------------------------------------------------ fitting core

def _make_pipe(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def _fit_core(X: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha: float, n_splits: int) -> dict[str, Any]:
    """Mean model, scale model and calibration factor c, using out-of-fold residuals."""
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("the context model needs at least 2 locations (GroupKFold by location)")
    splits = list(GroupKFold(n_splits=min(n_splits, n_groups)).split(X, y, groups))

    resid = np.full(len(y), np.nan)
    for train, held in splits:
        resid[held] = y[held] - _make_pipe(alpha).fit(X[train], y[train]).predict(X[held])
    log_abs = np.log(np.abs(resid) + RESIDUAL_FLOOR)

    log_scale = np.full(len(y), np.nan)
    for train, held in splits:
        log_scale[held] = _make_pipe(alpha).fit(X[train], log_abs[train]).predict(X[held])
    ratio = np.abs(resid) / np.exp(log_scale)
    c = float(np.median(ratio) / MEDIAN_ABS_Z)

    return {
        "mean": _make_pipe(alpha).fit(X, y),
        "scale": _make_pipe(alpha).fit(X, log_abs),
        "c": c,
        "oof_gap_rmse": float(np.sqrt(np.mean(resid ** 2))),
        "n_rows": int(len(y)),
        "n_locations": int(n_groups),
    }


def _outputs(mean: Pipeline, scale: Pipeline, c: float, sigma_min: float,
             features: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """expected, sigma, gap, z for the rows of df."""
    if len(df) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS, index=df.index, dtype="float64")
    pred = mean.predict(features)
    sigma = np.maximum(np.exp(scale.predict(features)) * c, sigma_min)
    expected = np.exp(pred) - EPS
    target = df["target"].to_numpy(dtype=float) if "target" in df else np.full(len(df), np.nan)
    gap = _log_pos(target) - np.log(expected + EPS)
    return pd.DataFrame({"expected": expected, "sigma": sigma, "gap": gap, "z": gap / sigma},
                        index=df.index)


def _location(df: pd.DataFrame, i: int) -> str:
    return str(df.attrs.get("location") or f"frame_{i}")


def _check_frames(frames: Sequence[pd.DataFrame]) -> None:
    if not frames:
        raise ValueError("no frames given")
    for i, df in enumerate(frames):
        if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("frames must be DataFrames with a DatetimeIndex")
        if "target" not in df:
            raise ValueError(f"{_location(df, i)}: no target column")
        if df.attrs.get("split") == "test":
            raise ValueError(f"{_location(df, i)} is a test location; training on it is not allowed")


def _resolve(value: Any, section: Mapping[str, Any], key: str) -> float:
    if value is not None:
        return float(value)
    if section.get(key) is not None:
        return float(section[key])
    return float(DEFAULTS[key])


# ------------------------------------------------------------------ model

class ContextModel:
    """Contract C3: fit / predict / oof_predict / save / load."""

    def __init__(
        self,
        alpha: float | None = None,
        sigma_min: float | None = None,
        test_year: int | None = None,
        context_vars: Sequence[str] | None = None,
        n_splits: int = 5,
        cfg: Mapping[str, Any] | None = None,
    ) -> None:
        if cfg is None and (alpha is None or sigma_min is None or test_year is None or context_vars is None):
            cfg = load_config()
        section = dict((cfg or {}).get("context_model") or {})
        self.alpha = _resolve(alpha, section, "alpha")
        self.sigma_min = _resolve(sigma_min, section, "sigma_min")
        self.test_year = int(test_year if test_year is not None else cfg["test_year"])
        self.context_vars = list(context_vars if context_vars is not None else cfg["context_vars"])
        self.n_splits = int(n_splits)
        self.feature_names_ = feature_names(self.context_vars)
        self.mean_: Pipeline | None = None
        self.scale_: Pipeline | None = None
        self.c_: float | None = None
        self.train_locations_: list[str] = []
        self.oof_folds_: dict[str, list[str]] = {}
        self.fit_info_: dict[str, Any] = {}

    # -- helpers
    def _features(self, df: pd.DataFrame) -> np.ndarray:
        return build_features(df, self.context_vars)[self.feature_names_].to_numpy(dtype=float)

    def _training_rows(self, frames: Sequence[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Xs, ys, gs = [], [], []
        for i, df in enumerate(frames):
            target = df["target"].to_numpy(dtype=float)
            keep = (df.index.year < self.test_year) & ~np.isnan(target)
            if not keep.any():
                continue
            sub = df.loc[keep]
            Xs.append(self._features(sub))
            ys.append(_log_pos(sub["target"].to_numpy(dtype=float)))
            gs.append(np.full(int(keep.sum()), _location(df, i), dtype=object))
        if not Xs:
            raise ValueError(f"no training rows with a target before {self.test_year}")
        return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)

    def _require_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None or self.c_ is None:
            raise RuntimeError("the context model is not fitted")

    # -- C3 API
    def fit(self, frames: Sequence[pd.DataFrame]) -> "ContextModel":
        """Fit on train-split frames, rows with a target before test_year only."""
        _check_frames(frames)
        X, y, groups = self._training_rows(frames)
        core = _fit_core(X, y, groups, self.alpha, self.n_splits)
        self.mean_, self.scale_, self.c_ = core["mean"], core["scale"], core["c"]
        self.train_locations_ = sorted({str(g) for g in groups})
        self.fit_info_ = {k: core[k] for k in ("c", "oof_gap_rmse", "n_rows", "n_locations")}
        log.info("context model fitted on %d rows from %d locations (c = %.3f)",
                 core["n_rows"], core["n_locations"], core["c"])
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame[expected, sigma, gap, z] on df.index (gap and z NaN where target is NaN)."""
        self._require_fitted()
        return _outputs(self.mean_, self.scale_, self.c_, self.sigma_min, self._features(df), df)

    def oof_predict(self, frames: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
        """One DataFrame[expected, sigma, gap, z] per frame, from a model that never saw that location."""
        _check_frames(frames)
        locations = [_location(df, i) for i, df in enumerate(frames)]
        unique = sorted(set(locations))
        if len(unique) < 3:
            raise ValueError("oof_predict needs at least 3 locations")
        outer = GroupKFold(n_splits=min(self.n_splits, len(unique)))
        dummy = np.zeros((len(frames), 1))
        outputs: list[pd.DataFrame | None] = [None] * len(frames)
        self.oof_folds_ = {}
        for train_idx, held_idx in outer.split(dummy, groups=np.array(locations, dtype=object)):
            train_frames = [frames[i] for i in train_idx]
            X, y, groups = self._training_rows(train_frames)
            core = _fit_core(X, y, groups, self.alpha, self.n_splits)
            seen = sorted({str(g) for g in groups})
            for i in held_idx:
                outputs[i] = _outputs(core["mean"], core["scale"], core["c"], self.sigma_min,
                                      self._features(frames[i]), frames[i])
                self.oof_folds_[locations[i]] = seen
        return [out for out in outputs if out is not None]

    def save(self, path: Path | str | None = None) -> Path:
        """Save to models/context_model.joblib (or `path`)."""
        self._require_fitted()
        path = Path(path) if path is not None else PATHS["models"] / DEFAULT_MODEL_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "version": MODEL_VERSION,
            "params": {"alpha": self.alpha, "sigma_min": self.sigma_min, "test_year": self.test_year,
                       "context_vars": self.context_vars, "n_splits": self.n_splits},
            "feature_names": self.feature_names_,
            "mean": self.mean_, "scale": self.scale_, "c": self.c_,
            "train_locations": self.train_locations_, "fit_info": self.fit_info_,
        }, path)
        return path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "ContextModel":
        """Load a model saved with save()."""
        path = Path(path) if path is not None else PATHS["models"] / DEFAULT_MODEL_NAME
        if not path.exists():
            raise FileNotFoundError("no context model; run: python -m src.detect.context_model --train")
        state = joblib.load(path)
        if state.get("version") != MODEL_VERSION:
            raise ValueError("saved context model has an unknown version; retrain it")
        model = cls(**state["params"])
        if model.feature_names_ != state["feature_names"]:
            raise ValueError("saved feature list does not match this code; retrain the model")
        model.mean_, model.scale_, model.c_ = state["mean"], state["scale"], state["c"]
        model.train_locations_ = list(state["train_locations"])
        model.fit_info_ = dict(state["fit_info"])
        return model


# ------------------------------------------------------------------ baseline

def _climatology_log(df: pd.DataFrame, window: int) -> np.ndarray:
    """Causal day-of-year mean of log(target + EPS) over the site's previous years (+/- window days)."""
    n = len(df)
    if n == 0:
        return np.empty(0)
    target = df["target"].to_numpy(dtype=float) if "target" in df else np.full(n, np.nan)
    years = df.index.year.to_numpy()
    doy = np.minimum(df.index.dayofyear.to_numpy(), 365) - 1
    uniq, yi = np.unique(years, return_inverse=True)
    grid = np.full((len(uniq), 365), np.nan)
    grid[yi, doy] = _log_pos(target)
    values = np.nan_to_num(grid, nan=0.0)
    counts = (~np.isnan(grid)).astype(float)
    sums = np.zeros_like(values)
    totals = np.zeros_like(counts)
    for offset in range(-window, window + 1):
        sums += np.roll(values, offset, axis=1)
        totals += np.roll(counts, offset, axis=1)
    past_sums = np.cumsum(sums, axis=0) - sums      # strictly earlier years only
    past_totals = np.cumsum(totals, axis=0) - totals
    with np.errstate(invalid="ignore", divide="ignore"):
        clim = np.where(past_totals > 0, past_sums / past_totals, np.nan)
    return clim[yi, doy]


class ClimatologyBaseline:
    """Comparison baseline with the C3 interface: causal day-of-year mean of the site's past years."""

    def __init__(self, window_days: int = 7, sigma_min: float | None = None,
                 test_year: int | None = None, cfg: Mapping[str, Any] | None = None) -> None:
        if cfg is None and (sigma_min is None or test_year is None):
            cfg = load_config()
        section = dict((cfg or {}).get("context_model") or {})
        self.window_days = int(window_days)
        self.sigma_min = _resolve(sigma_min, section, "sigma_min")
        self.test_year = int(test_year if test_year is not None else cfg["test_year"])
        self.sigma_: float | None = None

    def fit(self, frames: Sequence[pd.DataFrame]) -> "ClimatologyBaseline":
        """Set a constant sigma from robust residual spread on train years."""
        _check_frames(frames)
        resid = []
        for df in frames:
            before = df.index.year < self.test_year
            r = _log_pos(df["target"].to_numpy(dtype=float)) - _climatology_log(df, self.window_days)
            resid.append(r[before])
        r = np.concatenate(resid)
        r = r[~np.isnan(r)]
        if r.size == 0:
            raise ValueError("no residuals to fit the baseline (need at least 2 years per location)")
        spread = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        self.sigma_ = max(spread, self.sigma_min)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.sigma_ is None:
            raise RuntimeError("the baseline is not fitted")
        log_expected = _climatology_log(df, self.window_days)
        expected = np.exp(log_expected) - EPS
        target = df["target"].to_numpy(dtype=float) if "target" in df else np.full(len(df), np.nan)
        gap = _log_pos(target) - log_expected
        sigma = np.full(len(df), self.sigma_)
        return pd.DataFrame({"expected": expected, "sigma": sigma, "gap": gap, "z": gap / sigma},
                            index=df.index)

    def oof_predict(self, frames: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
        """Each site is predicted from its own past only, so this is out-of-fold by construction."""
        _check_frames(frames)
        return [self.predict(df) for df in frames]

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"version": MODEL_VERSION, "window_days": self.window_days, "sigma_min": self.sigma_min,
                     "test_year": self.test_year, "sigma": self.sigma_}, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "ClimatologyBaseline":
        state = joblib.load(Path(path))
        model = cls(window_days=state["window_days"], sigma_min=state["sigma_min"], test_year=state["test_year"])
        model.sigma_ = state["sigma"]
        return model


# ------------------------------------------------------------------ CLI

def _rmse(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def _zstats(z: np.ndarray) -> tuple[float, float]:
    z = z[~np.isnan(z)]
    return (float(np.mean(z)), float(np.std(z))) if z.size else (float("nan"), float("nan"))


def train_cli() -> int:
    """Train on all train locations, save, and print out-of-fold and test-year diagnostics."""
    from src.data.align import list_locations, load

    cfg = load_config()
    ids = list_locations("train", cfg)
    frames = [load(location_id, cfg) for location_id in ids]
    model = ContextModel(cfg=cfg).fit(frames)
    path = model.save()
    oof = model.oof_predict(frames)
    base = ClimatologyBaseline(cfg=cfg).fit(frames)
    ty = model.test_year

    print(f"{'location':<14}{'OOF rmse':>9}{'z mean':>8}{'z std':>7}{'clim rmse':>10}"
          f"   {ty} rmse{'z mean':>8}{'z std':>7}")
    pooled: dict[str, list[np.ndarray]] = {"oof_gap": [], "oof_z": [], "clim": [], "ty_gap": [], "ty_z": []}
    for location_id, df, o in zip(ids, frames, oof):
        before = df.index.year < ty
        in_ty = df.index.year == ty
        full = model.predict(df)
        clim = base.predict(df)
        parts = {
            "oof_gap": o["gap"].to_numpy()[before], "oof_z": o["z"].to_numpy()[before],
            "clim": clim["gap"].to_numpy()[before],
            "ty_gap": full["gap"].to_numpy()[in_ty], "ty_z": full["z"].to_numpy()[in_ty],
        }
        for key, value in parts.items():
            pooled[key].append(value)
        zm, zs = _zstats(parts["oof_z"])
        tm, ts = _zstats(parts["ty_z"])
        print(f"{location_id:<14}{_rmse(parts['oof_gap']):>9.3f}{zm:>8.2f}{zs:>7.2f}{_rmse(parts['clim']):>10.3f}"
              f"{_rmse(parts['ty_gap']):>12.3f}{tm:>8.2f}{ts:>7.2f}")
    p = {key: np.concatenate(value) for key, value in pooled.items()}
    zm, zs = _zstats(p["oof_z"])
    tm, ts = _zstats(p["ty_z"])
    print(f"{'ALL':<14}{_rmse(p['oof_gap']):>9.3f}{zm:>8.2f}{zs:>7.2f}{_rmse(p['clim']):>10.3f}"
          f"{_rmse(p['ty_gap']):>12.3f}{tm:>8.2f}{ts:>7.2f}")
    print(f"saved models/{path.name}; rows {model.fit_info_['n_rows']}, "
          f"locations {model.fit_info_['n_locations']}, c = {model.c_:.3f}")
    print("gap is in log units: a 12% step adds 0.113 to the gap")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Context model (expected value of the target).")
    parser.add_argument("--train", action="store_true", help="train on all train locations and save")
    args = parser.parse_args(argv)
    if not args.train:
        parser.print_help()
        return 0
    return train_cli()


if __name__ == "__main__":
    raise SystemExit(main())