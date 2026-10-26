"""Tests for src/detect/drift.py (offline, synthetic noise)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

import src.detect.drift as drift
from src.detect.step import count_episodes


def as_series(values, start: str = "2018-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D", name="date")
    return pd.Series(np.asarray(values, dtype=float), index=index)


def noisy(n: int, seed: int, noise: float = 0.1) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, noise, n)


def ramp_gap(n: int, seed: int, at: int, rise: float = 0.14, duration: int = 90, noise: float = 0.1) -> pd.Series:
    g = noisy(n, seed, noise)
    k = np.arange(n - at)
    g[at:] += rise * np.minimum((k + 1.0) / duration, 1.0)
    return as_series(g)


def step_gap(n: int, seed: int, at: int, size: float = 0.113, noise: float = 0.05) -> pd.Series:
    g = noisy(n, seed, noise)
    g[at:] += size
    return as_series(g)


# ---------------------------------------------------------------- contract

def test_output_contract():
    gap = ramp_gap(600, 1, 300, noise=0.05)
    out = drift.detect(gap / 0.05, gap)
    assert list(out.columns) == ["score", "alarm"]
    assert out.index.equals(gap.index)
    assert (out["score"] >= 0).all()
    assert out["alarm"].dtype == bool
    assert (out["alarm"] == (out["score"] >= 1.0)).all()
    assert isinstance(out.attrs["est_start"], pd.Timestamp)
    assert out.attrs["direction"] == "up"
    assert out.attrs["trigger"] in ("ewma", "slope")


def test_no_alarm_means_no_start():
    gap = as_series(np.zeros(300))
    out = drift.detect(gap, gap)
    assert not out["alarm"].any()
    assert out.attrs["est_start"] is None and out.attrs["direction"] is None


# ---------------------------------------------------------------- detection

def test_detects_ramp_before_day_120_and_estimates_start():
    delays, errors, early = [], [], 0
    for seed in range(20):
        gap = ramp_gap(800, seed, 400)
        out = drift.detect(gap / 0.1, gap)
        hits = np.flatnonzero(out["alarm"].to_numpy())
        if hits.size == 0:
            delays.append(np.inf)
            continue
        if hits[0] < 400:
            early += 1
            continue
        delays.append(hits[0] - 400)
        errors.append(abs((out.attrs["est_start"] - gap.index[400]).days))
    assert early <= 3
    assert np.mean(np.array(delays) < 120) >= 0.8
    assert np.median(errors) <= 20


def test_detects_negative_ramp():
    found = 0
    for seed in range(5):
        gap = ramp_gap(800, seed, 400, rise=-0.14)
        out = drift.detect(gap / 0.1, gap)
        hits = np.flatnonzero(out["alarm"].to_numpy())
        if hits.size and 400 <= hits[0] < 520 and out.attrs["direction"] == "down":
            found += 1
    assert found >= 4


def test_clean_five_years_default_params():
    def episodes(seed: int) -> int:
        gap = as_series(noisy(1826, seed))
        return count_episodes(drift.detect(gap / 0.1, gap)["alarm"])
    assert episodes(0) <= 1
    assert sum(episodes(seed) for seed in range(10)) <= 10


# ---------------------------------------------------------------- shape

def test_classify_shape_separates_step_and_ramp():
    steps = [drift.classify_shape(g, g.index[300], g.index[400])
             for g in (step_gap(600, s, 300) for s in range(10))]
    ramps = [drift.classify_shape(g, g.index[300], g.index[420])
             for g in (ramp_gap(600, s, 300, noise=0.05) for s in range(10))]
    assert steps.count("step") >= 9
    assert ramps.count("drift") >= 9


def test_classify_shape_long_drift_with_plateau():
    g = ramp_gap(800, 2, 300, duration=60, noise=0.05)
    assert drift.classify_shape(g, g.index[300], g.index[600]) == "drift"


def test_classify_shape_with_too_little_data():
    g = step_gap(20, 0, 10)
    assert drift.classify_shape(g, g.index[10], g.index[12], pre_days=2) == "step"


def test_fit_ramp_recovers_exact_shapes():
    t = np.arange(150, dtype=float)
    y = np.full(150, 0.01)
    y[50:] += 0.2 * np.minimum((t[50:] - 50 + 1.0) / 30.0, 1.0)
    fit = drift.fit_ramp(t, y)
    assert fit["best_tau"] == 50 and fit["best_D"] == 30 and fit["best_sse"] < 1e-12
    y2 = 0.01 + 0.1 * (t >= 70)
    fit2 = drift.fit_ramp(t, y2)
    assert fit2["best_tau"] == 70 and fit2["best_D"] == 1
    assert drift.fit_ramp(t[:5], y[:5]) is None


# ---------------------------------------------------------------- behaviour

def test_causality_prefix():
    gap = ramp_gap(700, 4, 350)
    z = gap / 0.1
    full = drift.detect(z, gap)
    for t in (50, 200, 380, 450, 699):
        part = drift.detect(z.iloc[:t], gap.iloc[:t])
        pd.testing.assert_frame_equal(part, full.iloc[:t])


def test_nan_days_never_start_an_alarm():
    gap = ramp_gap(700, 5, 300)
    gap.iloc[350:380] = np.nan
    out = drift.detect(gap / 0.1, gap)
    assert (out["score"].iloc[350:380] == out["score"].iloc[349]).all()
    alarm = out["alarm"].to_numpy()
    for i in np.flatnonzero(gap.isna().to_numpy()):
        assert alarm[i] == alarm[i - 1]
    empty = as_series(np.full(100, np.nan))
    out2 = drift.detect(empty, empty)
    assert (out2["score"] == 0).all() and not out2["alarm"].any()
    assert out2.attrs["est_start"] is None


def test_without_gap_uses_ewma_only():
    gap = ramp_gap(700, 6, 300, noise=0.05)
    out = drift.detect(gap / 0.05)
    assert list(out.columns) == ["score", "alarm"]
    assert out["alarm"].iloc[300:].any()


def test_rolling_slope_t():
    y = 0.002 * np.arange(100) + noisy(100, 8, 0.01)
    slope, t = drift.rolling_slope_t(y, 60, 30)
    assert np.isnan(slope[:29]).all()
    assert slope[-1] == pytest.approx(0.002, abs=0.0005)
    assert t[-1] > 10


def test_speed_seven_years():
    gap = as_series(noisy(2557, 9))
    z = gap / 0.1
    best = min(_timed(lambda: drift.detect(z, gap)) for _ in range(3))
    assert best < 0.2


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def test_params():
    p = drift.merge_params({"lambda": None, "L": 3.0})
    assert p["lambda"] == drift.DEFAULTS["lambda"] and p["L"] == 3.0
    with pytest.raises(ValueError):
        drift.merge_params({"lambda": 0})
    with pytest.raises(ValueError):
        drift.merge_params({"slope_window": 5})
    cfg = {"drift": {"lambda": 0.1}, "step": {"phi": 0.3}}
    assert drift.params_from_config(cfg)["phi"] == 0.3