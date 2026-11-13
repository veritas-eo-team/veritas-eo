"""Compare context-model variants on calibrated gap noise.

Training sites: out-of-fold (GroupKFold by location), training years only.
Test sites: model fit on all training rows, seasonal calibration on test_year - 1,
noise measured in test_year. Lower std = quieter gap = small attacks stand out sooner.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from src.common import load_config
from src.data.align import list_locations, load
from src.detect.context_model import _harmonics, _log_pos, _make_pipe, build_features

VARIANTS = {
    "ridge (current)": ("base", "ridge"),
    "ridge + nb extras": ("extra", "ridge"),
    "boosting + nb extras": ("extra", "hgb"),
}


def features(df, context_vars, kind):
    X = build_features(df, context_vars)
    if kind == "base":
        return X
    nb_cols = sorted(c for c in df.columns if c.startswith("nb_"))
    nb = _log_pos(df[nb_cols].to_numpy(dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        X["nb_log_min"] = np.nanmin(nb, axis=1)
        X["nb_log_max"] = np.nanmax(nb, axis=1)
    X["nb_clearness"] = X["nb_log_mean"] - X["log_clrsky"]
    X["nb_range"] = X["nb_log_max"] - X["nb_log_min"]
    return X


def make_model(kind, alpha):
    if kind == "ridge":
        return _make_pipe(alpha)
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def calibrated_std(index, resid, ref_mask, eval_mask):
    """Std of the residual after removing an annual cycle fitted on the reference rows."""
    ok = ref_mask & ~np.isnan(resid)
    if ok.sum() < 30:
        return np.nan
    H = _harmonics(index)
    coef, *_ = np.linalg.lstsq(H[ok], resid[ok], rcond=None)
    e = (resid - H @ coef)[eval_mask]
    e = e[~np.isnan(e)]
    return float(np.std(e)) if e.size else np.nan


def rows_of(df, context_vars, feat_kind, years_mask):
    y = _log_pos(df["target"].to_numpy(dtype=float))
    keep = years_mask & ~np.isnan(y)
    X = features(df, context_vars, feat_kind).to_numpy(dtype=float)
    return X, y, keep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    cfg = load_config()
    cv = cfg["context_vars"]
    ty = int(cfg["test_year"])
    alpha = float((cfg.get("context_model") or {}).get("alpha") or 1.0)

    train_ids = list_locations("train", cfg)
    test_ids = list_locations("test", cfg)
    train = {i: load(i, cfg) for i in train_ids}
    test = {i: load(i, cfg) for i in test_ids}

    oof_table = {i: {} for i in train_ids}
    test_table = {i: {} for i in test_ids}

    for name, (feat_kind, model_kind) in VARIANTS.items():
        Xs, ys, gs, spans = [], [], [], {}
        start = 0
        for i, df in train.items():
            X, y, keep = rows_of(df, cv, feat_kind, df.index.year < ty)
            Xs.append(X[keep]); ys.append(y[keep]); gs.append(np.full(int(keep.sum()), i, dtype=object))
            spans[i] = (start, start + int(keep.sum()), keep)
            start += int(keep.sum())
        X_all, y_all, g_all = np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)

        oof = np.full(len(y_all), np.nan)
        for tr, held in GroupKFold(n_splits=5).split(X_all, y_all, g_all):
            oof[held] = y_all[held] - make_model(model_kind, alpha).fit(X_all[tr], y_all[tr]).predict(X_all[held])

        for i, df in train.items():
            a, b, keep = spans[i]
            resid = np.full(len(df), np.nan)
            resid[keep] = oof[a:b]
            years = df.index.year
            first = years.min()
            oof_table[i][name] = calibrated_std(df.index, resid, years == first, (years > first) & (years < ty))

        full = make_model(model_kind, alpha).fit(X_all, y_all)
        for i, df in test.items():
            X, y, _ = rows_of(df, cv, feat_kind, np.ones(len(df), bool))
            resid = y - full.predict(X)
            years = df.index.year
            test_table[i][name] = calibrated_std(df.index, resid, years == ty - 1, years == ty)
        print(f"done: {name}")

    for title, table in (("Training sites, out-of-fold (calibrated gap std):", oof_table),
                         (f"Test sites, {ty} (calibrated gap std):", test_table)):
        df = pd.DataFrame(table).T.round(4)
        df.loc["MEAN"] = df.mean().round(4)
        base = df["ridge (current)"]
        for col in df.columns[1:]:
            df[f"{col} vs current"] = ((df[col] / base - 1) * 100).round(1).astype(str) + "%"
        print("\n" + title)
        print(df.to_string())


if __name__ == "__main__":
    main()