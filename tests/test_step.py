"""Tests for src/detect/step.py (offline, synthetic noise)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

import src.detect.step as step

TRIAL_PARAMS = {"k": 0.3, "h": 10.0, "phi": 0.0}


def as_series(values, start: str = "2018-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D", name="date")
    return pd.Series(np.asarray(values, dtype=float), index=index)


def noise(n: int, seed: int, shift: float = 0.0, at: int | None = None) -> pd.Series:
    x = np.random.default_rng(seed).normal(0.0, 1.0, n)
    if at is not None:
        x[at:] += shift
    return as_series(x)


def ar1(n: int, phi: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, np.sqrt(1.0 - phi ** 2), n)
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + shocks[t]
    return as_series(x)


def step_trials(shift: float):
    """20 seeds of N(0,1) with `shift` added from day 500."""
    delays, errors, sizes, early, directions = [], [], [], 0, []
    for seed in range(20):
        z = noise(1000, seed, shift, 500)
        out = step.detect(z, z * 0.1, TRIAL_PARAMS)
        hits = np.flatnonzero(out["alarm"].to_numpy())
        if hits.size == 0:
            delays.append(np.inf)
            continue
        if hits[0] < 500:
            early += 1
            continue
        delays.append(hits[0] - 500)
        errors.append(abs((out.attrs["est_start"] - z.index[500]).days))
        sizes.append(out.attrs["size_pct"])
        directions.append(out.attrs["direction"])
    return np.array(delays), np.array(errors), np.array(sizes), early, directions


# ---------------------------------------------------------------- contract

def test_output_contract():
    z = noise(500, 1, 1.5, 250)
    out = step.detect(z, z * 0.1, TRIAL_PARAMS)
    assert list(out.columns) == ["score", "alarm"]
    assert out.index.equals(z.index)
    assert (out["score"] >= 0).all()
    assert out["alarm"].dtype == bool
    assert (out["alarm"] == (out["score"] >= 1.0)).all()
    assert isinstance(out.attrs["est_start"], pd.Timestamp)


def test_no_alarm_means_no_start():
    out = step.detect(as_series(np.zeros(300)))
    assert not out["alarm"].any()
    assert out.attrs["est_start"] is None and out.attrs["direction"] is None


# ---------------------------------------------------------------- detection

def test_detects_positive_step():
    delays, errors, sizes, early, directions = step_trials(+0.6)
    assert early <= 8
    assert np.median(delays) <= 40
    assert np.median(errors) <= 10
    assert 0.03 < np.median(sizes) < 0.15
    assert set(directions) == {"up"}


def test_detects_negative_step():
    delays, errors, sizes, early, directions = step_trials(-0.6)
    assert early <= 8
    assert np.median(delays) <= 40
    assert np.median(errors) <= 10
    assert -0.15 < np.median(sizes) < -0.03
    assert set(directions) == {"down"}


def test_clean_five_years_default_params():
    assert step.count_episodes(step.detect(noise(1826, 0))["alarm"]) <= 1
    total = sum(step.count_episodes(step.detect(noise(1826, seed))["alarm"]) for seed in range(10))
    assert total <= 10


def test_prewhitening_reduces_false_alarms_on_autocorrelated_noise():
    raw = white = 0
    for seed in range(5):
        z = ar1(1826, 0.5, seed)
        raw += int(step.detect(z, params={"phi": 0.0})["alarm"].sum())
        white += int(step.detect(z, params={"phi": 0.5})["alarm"].sum())
    assert white < raw


# ---------------------------------------------------------------- behaviour

def test_causality_prefix():
    z = noise(800, 3, 1.0, 400)
    gap = z * 0.1
    full = step.detect(z, gap, TRIAL_PARAMS)
    for t in (100, 420, 450, 799):
        part = step.detect(z.iloc[:t], gap.iloc[:t], TRIAL_PARAMS)
        pd.testing.assert_frame_equal(part, full.iloc[:t])


def test_nan_days_carry_the_statistic():
    z = noise(600, 4, 1.0, 200)
    z.iloc[300:320] = np.nan
    out = step.detect(z, params=TRIAL_PARAMS)
    assert (out["score"].iloc[300:320] == out["score"].iloc[299]).all()
    alarm = out["alarm"].to_numpy()
    for i in np.flatnonzero(z.isna().to_numpy()):
        assert alarm[i] == alarm[i - 1]  # a NaN day never starts an alarm
    empty = step.detect(as_series(np.full(100, np.nan)))
    assert (empty["score"] == 0).all() and not empty["alarm"].any()
    assert empty.attrs["est_start"] is None


def test_score_is_capped():
    out = step.detect(noise(400, 5, 5.0, 100))
    assert out["score"].max() <= 2.0 + 1e-12


def test_speed_seven_years():
    z = noise(2557, 6)
    best = min(_timed(lambda: step.detect(z, z * 0.1)) for _ in range(3))
    assert best < 0.05


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# ---------------------------------------------------------------- helpers

def test_refine_start_finds_the_step():
    rng = np.random.default_rng(7)
    gap = as_series(rng.normal(0.0, 0.01, 120))
    gap.iloc[60:] += 0.1
    start, size = step.refine_start(gap, gap.index[100])
    assert abs((start - gap.index[60]).days) <= 1
    assert size == pytest.approx(np.exp(0.1) - 1.0, abs=0.005)
    assert step.refine_start(gap.iloc[:5], gap.index[4]) == (None, pytest.approx(float("nan"), nan_ok=True))


def test_estimate_phi():
    series = [ar1(2000, 0.5, seed) for seed in range(3)]
    series[0].iloc[100:150] = np.nan
    assert step.estimate_phi(series) == pytest.approx(0.5, abs=0.05)
    assert step.estimate_phi([as_series(np.full(10, np.nan))]) == 0.0


def test_params():
    p = step.merge_params({"k": None, "h": 5})
    assert p["k"] == step.DEFAULTS["k"] and p["h"] == 5.0
    with pytest.raises(ValueError):
        step.merge_params({"h": 0})
    with pytest.raises(ValueError):
        step.merge_params({"phi": 1.0})


def test_count_episodes():
    assert step.count_episodes([False, True, True, False, True]) == 2
    assert step.count_episodes([True, True]) == 1
    assert step.count_episodes([]) == 0