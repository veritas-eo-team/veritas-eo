"""Tests for src/attacks/inject.py (offline, synthetic data)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import src.attacks.inject as inj
from src.synthetic import make_synthetic_frame

START = pd.Timestamp("2024-03-01")
DAY = pd.Timedelta(days=1)


@pytest.fixture
def series():
    return make_synthetic_frame(seed=3)["target"]


def pos(s: pd.Series, when) -> int:
    return int(s.index.get_loc(pd.Timestamp(when)))


def outside_unchanged(clean: pd.Series, attacked: pd.Series, mask: pd.Series) -> bool:
    return np.array_equal(clean[~mask].to_numpy(), attacked[~mask].to_numpy(), equal_nan=True)


# ---------------------------------------------------------------- each attack

def test_step_ratio_exact(series):
    attacked, labels = inj.inject(series, "step", start=START)
    ratio = attacked / series
    assert np.allclose(ratio[ratio.index >= START], 1.12, rtol=1e-12)
    assert (ratio[ratio.index < START] == 1.0).all()
    assert labels["type"] == "step" and labels["size"] == pytest.approx(0.12)
    assert labels["start"] == START and labels["end"] == series.index[-1]


def test_negative_step(series):
    attacked, _ = inj.inject(series, "step", size=-0.12, start=START)
    ratio = attacked / series
    assert np.allclose(ratio[ratio.index >= START], 0.88, rtol=1e-12)


def test_drift_reaches_size_and_holds(series):
    attacked, labels = inj.inject(series, "drift", start=START)
    r = (attacked / series).to_numpy()
    s = pos(series, START)
    assert np.allclose(r[:s], 1.0)
    k = np.arange(60)
    assert np.allclose(r[s:s + 60], 1.0 + 0.15 * (k + 1) / 60)
    assert r[s + 59] == pytest.approx(1.15)
    assert np.allclose(r[s + 60:], 1.15)
    assert labels["end"] == series.index[-1]
    assert labels["params"]["duration"] == 60


@pytest.mark.parametrize("source", ["lastyear", "recent", "random"])
def test_replay_copies_source_window(series, source):
    attacked, labels = inj.inject(series, "replay", start=START, source=source, seed=4)
    s = pos(series, START)
    src = pos(series, labels["source_start"])
    np.testing.assert_array_equal(attacked.iloc[s:s + 7].to_numpy(), series.iloc[src:src + 7].to_numpy())
    assert labels["end"] == START + 6 * DAY
    if source == "lastyear":
        assert labels["source_start"] == START - 365 * DAY
    elif source == "recent":
        assert labels["source_start"] == START - 7 * DAY
    else:
        assert labels["source_start"] + 6 * DAY < START
    mask = inj.attack_mask(labels, series.index)
    assert outside_unchanged(series, attacked, mask)


def test_replay_noise_is_seeded(series):
    a1, l1 = inj.inject(series, "replay", start=START, noise=0.01, seed=5)
    a2, l2 = inj.inject(series, "replay", start=START, noise=0.01, seed=5)
    pd.testing.assert_series_equal(a1, a2)
    s, src = pos(series, START), pos(series, l1["source_start"])
    rel = a1.iloc[s:s + 7].to_numpy() / series.iloc[src:src + 7].to_numpy() - 1.0
    assert np.any(rel != 0) and np.all(np.abs(rel) < 0.1)


def test_spike(series):
    attacked, labels = inj.inject(series, "spike", start=START)
    changed = attacked[attacked != series]
    assert list(changed.index) == [START]
    assert attacked[START] / series[START] == pytest.approx(1.5)
    assert labels["end"] == START


def test_stuck_holds_start_value(series):
    attacked, labels = inj.inject(series, "stuck", start=START)
    s = pos(series, START)
    assert (attacked.iloc[s:s + 7] == series.iloc[s]).all()
    assert labels["end"] == START + 6 * DAY
    assert labels["params"]["held_value"] == pytest.approx(series.iloc[s])


def test_stuck_with_missing_start_uses_last_valid(series):
    s2 = series.copy()
    s2[START] = np.nan
    attacked, labels = inj.inject(s2, "stuck", start=START)
    assert np.isnan(attacked[START])
    prev = s2[START - DAY]
    assert (attacked[START + DAY: START + 6 * DAY] == prev).all()


def test_none_changes_nothing(series):
    attacked, labels = inj.inject(series, "none")
    pd.testing.assert_series_equal(attacked, series)
    assert labels["type"] == "none" and labels["start"] is None
    assert not inj.attack_mask(labels, series.index).any()


# ---------------------------------------------------------------- general rules

@pytest.mark.parametrize("attack,n_days", [("replay", 7), ("spike", 1), ("stuck", 7)])
def test_mask_matches_labels_and_outside_unchanged(series, attack, n_days):
    attacked, labels = inj.inject(series, attack, start=START)
    mask = inj.attack_mask(labels, series.index)
    assert int(mask.sum()) == n_days
    assert mask[mask].index[0] == labels["start"] and mask[mask].index[-1] == labels["end"]
    assert outside_unchanged(series, attacked, mask)
    assert (attacked[mask] != series[mask]).any()


@pytest.mark.parametrize("attack", ["step", "drift"])
def test_open_ended_mask(series, attack):
    attacked, labels = inj.inject(series, attack, start=START)
    mask = inj.attack_mask(labels, series.index)
    assert int(mask.sum()) == len(series) - pos(series, START)
    assert outside_unchanged(series, attacked, mask)


@pytest.mark.parametrize("attack", ["step", "drift", "replay", "spike", "stuck"])
def test_nan_stays_nan(attack):
    s = make_synthetic_frame(seed=3, missing_frac=0.1)["target"]
    attacked, _ = inj.inject(s, attack, start=START)
    assert attacked[s.isna()].isna().all()


def test_returns_copy(series):
    before = series.copy()
    inj.inject(series, "step", start=START)
    inj.inject(series, "replay", start=START, noise=0.01)
    pd.testing.assert_series_equal(series, before)


def test_same_seed_same_output(series):
    a1, l1 = inj.inject(series, "replay", seed=7, source="random", noise=0.01)
    a2, l2 = inj.inject(series, "replay", seed=7, source="random", noise=0.01)
    pd.testing.assert_series_equal(a1, a2)
    assert l1 == l2
    starts = {inj.inject(series, "step", seed=k)[1]["start"] for k in range(5)}
    assert len(starts) > 1


def test_random_start_stays_in_window(series):
    for seed in range(20):
        _, labels = inj.inject(series, "replay", seed=seed,
                               window_start="2024-02-01", window_end="2024-09-30")
        assert pd.Timestamp("2024-02-01") <= labels["start"] <= pd.Timestamp("2024-09-30")
        assert labels["params"]["start_mode"] == "random"
    for seed in range(20):
        _, labels = inj.inject(series, "stuck", seed=seed, window_start="2024-12-01")
        assert labels["end"] <= series.index[-1]


def test_decimals_rounds_only_attacked_days(series):
    attacked, labels = inj.inject(series, "step", start=START, decimals=2)
    window = attacked[attacked.index >= START]
    assert (window == window.round(2)).all()
    mask = inj.attack_mask(labels, series.index)
    assert outside_unchanged(series, attacked, mask)


def test_labels_types_and_json(series):
    _, labels = inj.inject(series, "replay", start=START, noise=0.005)
    assert isinstance(labels["start"], pd.Timestamp) and isinstance(labels["end"], pd.Timestamp)
    assert isinstance(labels["source_start"], pd.Timestamp)
    json.dumps(labels["params"])
    assert set(labels) >= {"type", "start", "end", "size", "params"}


# ---------------------------------------------------------------- errors

def test_bad_inputs(series):
    with pytest.raises(ValueError):
        inj.inject(series, "flood")
    with pytest.raises(ValueError):
        inj.inject(series.iloc[::2], "step", start=START)          # not daily
    with pytest.raises(ValueError):
        inj.inject(series, "replay", start=START, noise=-0.1)
    with pytest.raises(ValueError):
        inj.inject(series, "step", size=-1.5, start=START)
    with pytest.raises(ValueError):
        inj.inject(series, "replay", start=START, source="tomorrow")


def test_replay_needs_history(series):
    with pytest.raises(ValueError):
        inj.inject(series, "replay", start=series.index[100])       # lastyear needs 365 days


def test_attack_must_fit(series):
    with pytest.raises(ValueError):
        inj.inject(series, "stuck", start=series.index[-3])


# ---------------------------------------------------------------- smart attacks and demo

def test_adaptive_step(series):
    attacked, labels = inj.adaptive_step(series, 0.05, start=START)
    assert labels["type"] == "step"
    assert labels["size"] == pytest.approx(0.045)
    assert labels["params"]["adaptive"] is True and labels["params"]["limit"] == 0.05
    ratio = attacked / series
    assert np.allclose(ratio[ratio.index >= START], 1.045)


def test_slow_ramp(series):
    start = pd.Timestamp("2023-01-01")
    attacked, labels = inj.slow_ramp(series, 0.15, 365, start=start)
    assert labels["type"] == "drift" and labels["params"]["slow"] is True
    r = (attacked / series).to_numpy()
    s = pos(series, start)
    assert r[s + 364] == pytest.approx(1.15)
    assert r[s + 100] == pytest.approx(1.0 + 0.15 * 101 / 365)


def test_demo_writes_png(tmp_path):
    path = inj.demo(out_path=tmp_path / "examples.png", location="no_such_place")
    assert path.exists() and path.stat().st_size > 0