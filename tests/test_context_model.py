"""Tests for src/detect/context_model.py (offline, synthetic frames)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.detect.context_model as cm
from src.common import EPS
from src.synthetic import make_synthetic_frame

CONTEXT_VARS = ["CLRSKY_SFC_SW_DWN", "T2M", "RH2M", "PRECTOTCORR", "GWETTOP"]
TEST_YEAR = 2024


def new_model(**overrides) -> cm.ContextModel:
    params = dict(alpha=1.0, sigma_min=0.02, test_year=TEST_YEAR, context_vars=CONTEXT_VARS, n_splits=5)
    params.update(overrides)
    return cm.ContextModel(**params)


@pytest.fixture(scope="module")
def frames():
    return [make_synthetic_frame(location=f"syn{i}", seed=i) for i in range(4)]


@pytest.fixture(scope="module")
def fitted(frames):
    return new_model().fit(frames[:3])


def test_predict_columns_and_index(fitted, frames):
    out = fitted.predict(frames[3])
    assert list(out.columns) == cm.OUTPUT_COLUMNS
    assert out.index.equals(frames[3].index)
    assert (out["sigma"] >= 0.02).all()
    assert out["expected"].notna().all()


def test_clean_z_is_standardised(frames):
    oof = new_model().oof_predict(frames)
    z = pd.concat([o["z"] for o in oof]).dropna().to_numpy()
    assert abs(z.mean()) < 0.3
    assert 0.7 < z.std() < 1.5
    assert abs(np.median(np.abs(z)) - 0.674) < 0.2


def test_step_shifts_gap_by_0113_and_target_is_not_a_feature(fitted, frames):
    df = frames[3]
    clean = fitted.predict(df)
    attacked = df.copy()
    attacked.loc["2024-03-01":, "target"] *= 1.12
    out = fitted.predict(attacked)
    pd.testing.assert_series_equal(out["expected"], clean["expected"])  # expected ignores the target
    pd.testing.assert_series_equal(out["sigma"], clean["sigma"])
    shift = (out["gap"] - clean["gap"]).loc["2024-03-01":].mean()
    assert shift == pytest.approx(0.113, abs=0.003)
    assert (out["gap"] - clean["gap"]).loc[:"2024-02-29"].abs().max() == 0


def test_fit_refuses_test_split(frames):
    test_frame = make_synthetic_frame(location="hidden", seed=9, split="test")
    with pytest.raises(ValueError):
        new_model().fit(frames[:2] + [test_frame])
    with pytest.raises(ValueError):
        new_model().oof_predict(frames[:3] + [test_frame])


def test_fit_ignores_test_year_rows(fitted, frames):
    altered = []
    for df in frames[:3]:
        copy = df.copy()
        copy.loc[f"{TEST_YEAR}-01-01":, "target"] = 999.0
        altered.append(copy)
    other = new_model().fit(altered)
    np.testing.assert_allclose(other.predict(frames[3])["expected"], fitted.predict(frames[3])["expected"])


def test_save_load_round_trip(fitted, frames, tmp_path):
    path = fitted.save(tmp_path / "model.joblib")
    loaded = cm.ContextModel.load(path)
    pd.testing.assert_frame_equal(loaded.predict(frames[3]), fitted.predict(frames[3]))
    assert loaded.c_ == fitted.c_


def test_oof_is_out_of_fold_by_location(frames):
    model = new_model()
    oof = model.oof_predict(frames)
    assert len(oof) == len(frames)
    for out, df in zip(oof, frames):
        assert out.index.equals(df.index)
    for location, seen in model.oof_folds_.items():
        assert location not in seen
    # with 4 locations each fold leaves exactly one out: equal to a fit on the other three
    reference = new_model().fit(frames[1:]).predict(frames[0])
    np.testing.assert_allclose(oof[0]["expected"], reference["expected"])
    np.testing.assert_allclose(oof[0]["sigma"], reference["sigma"])


def test_fixed_feature_width(frames):
    full = frames[0]
    two = full.drop(columns=["nb_2", "nb_3"])
    bare = full.drop(columns=["nb_0", "nb_1", "nb_2", "nb_3", "ndvi", "cloud", "days_since_clear",
                              "ctx_CLRSKY_SFC_SW_DWN"])
    names = cm.feature_names(CONTEXT_VARS)
    for df in (full, two, bare):
        assert list(cm.build_features(df, CONTEXT_VARS).columns) == names
    assert "target" not in " ".join(names)


def test_predict_without_neighbours_or_satellite(fitted, frames):
    bare = frames[3].drop(columns=["nb_0", "nb_1", "nb_2", "nb_3", "ndvi", "cloud", "days_since_clear"])
    out = fitted.predict(bare)
    assert np.isfinite(out["expected"]).all()


def test_gap_nan_where_target_missing(fitted):
    df = make_synthetic_frame(location="holes", seed=11, missing_frac=0.1)
    out = fitted.predict(df)
    missing = df["target"].isna()
    assert out.loc[missing, "gap"].isna().all() and out.loc[missing, "z"].isna().all()
    assert out["expected"].notna().all()


def test_needs_two_locations(frames):
    with pytest.raises(ValueError):
        new_model().fit([frames[0]])


def test_climatology_baseline_interface_and_causality(frames):
    base = cm.ClimatologyBaseline(sigma_min=0.02, test_year=TEST_YEAR).fit(frames[:3])
    df = frames[3]
    out = base.predict(df)
    assert list(out.columns) == cm.OUTPUT_COLUMNS
    assert out.loc[:"2018-12-31", "expected"].isna().all()   # no past year yet
    assert out.loc["2019-01-01":, "expected"].notna().all()
    prefix = base.predict(df.iloc[:1500])
    pd.testing.assert_frame_equal(prefix, out.iloc[:1500])
    assert len(base.oof_predict(frames)) == len(frames)


def test_context_model_beats_climatology(frames):
    oof = new_model().oof_predict(frames)
    base = cm.ClimatologyBaseline(sigma_min=0.02, test_year=TEST_YEAR).fit(frames)
    years = slice("2019-01-01", "2023-12-31")
    ctx = np.concatenate([o.loc[years, "gap"].to_numpy() for o in oof])
    clim = np.concatenate([base.predict(df).loc[years, "gap"].to_numpy() for df in frames])
    rmse = lambda x: np.sqrt(np.nanmean(x ** 2))
    assert rmse(ctx) < 0.7 * rmse(clim)



def _gap_frame(n: int = 900, bias: float = 0.05, step_at: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    index = pd.date_range("2020-01-01", periods=n, freq="D", name="date")
    gap = bias + rng.normal(0.0, 0.05, n)
    if step_at is not None:
        gap[step_at:] += 0.113
    sigma = np.full(n, 0.05)
    return pd.DataFrame({"expected": np.full(n, 5.0), "sigma": sigma, "gap": gap, "z": gap / sigma},
                        index=index)


def test_calibration_removes_site_offset():
    cal = cm.calibrate_site(_gap_frame())
    assert cal["gap"].iloc[:30].isna().all()             # not enough reference days yet
    assert abs(cal["gap"].iloc[365:].mean()) < 0.01
    assert cal.attrs["site_offset"] == pytest.approx(0.05, abs=0.01)
    assert 0.8 < cal["z"].iloc[365:].std() < 1.2


def test_calibration_keeps_a_later_step():
    cal = cm.calibrate_site(_gap_frame(step_at=500))
    assert cal["gap"].iloc[500:].mean() == pytest.approx(0.113, abs=0.01)


def test_calibration_is_causal():
    out = _gap_frame()
    full = cm.calibrate_site(out)
    for t in (10, 200, 400, 800):
        pd.testing.assert_frame_equal(cm.calibrate_site(out.iloc[:t]), full.iloc[:t], check_freq=False)


def test_calibrated_output_keeps_c3_identities(fitted, frames):
    df = frames[3]
    cal = cm.calibrate_site(fitted.predict(df))
    ok = cal["gap"].notna()
    lhs = np.log(df["target"][ok] + EPS) - np.log(cal["expected"][ok] + EPS)
    np.testing.assert_allclose(cal["gap"][ok], lhs, atol=1e-9)
    np.testing.assert_allclose(cal["z"][ok], cal["gap"][ok] / cal["sigma"][ok])