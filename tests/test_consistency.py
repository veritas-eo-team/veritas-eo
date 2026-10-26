"""Tests for src/detect/consistency.py (offline, synthetic data)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import src.detect.consistency as cons

CLR = cons.CLRSKY_COL


def clear_frame(n: int = 730, seed: int = 0, level: float = 0.97) -> pd.DataFrame:
    """Clear days sit just below clear-sky (clearness up to 0.99), no Sentinel-2 days."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=n, freq="D", name="date")
    doy = index.day_of_year.to_numpy(dtype=float)
    clr = 7.0 + 2.0 * np.cos(2.0 * np.pi * (doy - 172.0) / 365.25)
    clearness = np.clip(level - rng.exponential(0.1, n), 0.2, 0.99)
    return pd.DataFrame({"target": np.round(clr * clearness, 2), CLR: np.round(clr, 2), "cloud": np.nan},
                        index=index)


def stepped(df: pd.DataFrame, at: int = 365, size: float = 0.12) -> pd.DataFrame:
    out = df.copy()
    col = out.columns.get_loc("target")
    out.iloc[at:, col] = out["target"].to_numpy()[at:] * (1.0 + size)
    return out


def test_output_contract():
    out = cons.detect(clear_frame())
    assert list(out.columns) == ["score", "alarm", "over_clearsky", "cloud_conflict"]
    for col in ("alarm", "over_clearsky", "cloud_conflict"):
        assert out[col].dtype == bool
    assert (out["alarm"] == (out["score"] >= 1.0)).all()


def test_clean_data_stays_quiet():
    out = cons.detect(clear_frame())
    assert not out["over_clearsky"].any()
    assert not out["alarm"].any()
    assert (out["score"] == 0).all()
    assert out.attrs["est_start"] is None


def test_step_raises_over_clearsky_alarms():
    df = stepped(clear_frame())
    out = cons.detect(df)
    assert not out["alarm"].iloc[:365].any()
    assert out["alarm"].iloc[365:379].any()
    assert out["over_clearsky"].iloc[365:].mean() > 0.2
    assert df.index[365] <= out.attrs["est_start"] <= df.index[378]


def test_cloud_conflict():
    df = clear_frame()
    t_col, c_col = df.columns.get_loc("target"), df.columns.get_loc("cloud")
    clr = df[CLR].to_numpy()
    for d in (100, 105):                       # bright readings under heavy cloud
        df.iloc[d, t_col] = round(0.95 * clr[d], 2)
        df.iloc[d, c_col] = 0.9
    df.iloc[200, t_col] = round(0.3 * clr[200], 2)  # dark under heavy cloud: consistent
    df.iloc[200, c_col] = 0.9
    out = cons.detect(df)
    assert out["cloud_conflict"].iloc[[100, 105]].all()
    assert not out["cloud_conflict"].iloc[200]
    assert not out["alarm"].iloc[100] and out["alarm"].iloc[105]
    assert out.attrs["est_start"] == df.index[100]


def test_missing_columns_give_zero_score():
    for df in (clear_frame()[["target"]], clear_frame().drop(columns=["cloud"])):
        out = cons.detect(df)
        assert (out["score"] == 0).all() and not out["alarm"].any()
    nan_clr = stepped(clear_frame())
    nan_clr[CLR] = np.nan
    out = cons.detect(nan_clr)
    assert not out["over_clearsky"].any() and not out["alarm"].any()


def test_causality_prefix():
    df = stepped(clear_frame())
    full = cons.detect(df)
    for t in (100, 366, 370, 700):
        pd.testing.assert_frame_equal(cons.detect(df.iloc[:t]), full.iloc[:t])


def test_natural_exceedance_rate_and_tolerance_choice():
    frames = []
    for seed in range(3):
        df = clear_frame(seed=seed)
        rows = np.random.default_rng(seed).choice(len(df), size=len(df) // 10, replace=False)
        col = df.columns.get_loc("target")
        df.iloc[rows, col] = np.round(df[CLR].to_numpy()[rows] * 1.015, 2)  # natural 1.5% excess
        frames.append(df)
    rates = cons.natural_exceedance_rate(frames)
    assert rates.loc[0.01, "rate"] > 0.05
    assert rates.loc[0.02, "rate"] == 0.0
    assert cons.choose_clearsky_tol(frames, max_rate=0.005) == 0.02