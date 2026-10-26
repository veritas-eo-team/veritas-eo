"""Tests for src/detect/replay.py (offline, synthetic data)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

import src.detect.replay as replay
from src.attacks.inject import attack_mask, inject
from src.synthetic import make_synthetic_frame

DAY = pd.Timedelta(days=1)
START = "2024-03-01"


@pytest.fixture(scope="module")
def target():
    return make_synthetic_frame(seed=2)["target"]


def clean_z(index: pd.DatetimeIndex, seed: int = 0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(0.0, 1.0, len(index)), index=index)


def shifted_replay(target: pd.Series):
    attacked, labels = inject(target, "replay", start=START, noise=0.01, seed=3)
    z = clean_z(target.index, 1)
    z[attack_mask(labels, target.index)] += 3.0
    return attacked, labels, z


def test_output_contract(target):
    out = replay.detect(target, clean_z(target.index))
    assert list(out.columns) == ["score", "alarm", "exact", "stuck"]
    assert out.index.equals(target.index)
    for col in ("alarm", "exact", "stuck"):
        assert out[col].dtype == bool
    assert (out["score"] >= 0).all()
    assert (out["alarm"] == (out["score"] >= 1.0)).all()


def test_exact_copy_alarms_on_last_day(target):
    attacked, labels = inject(target, "replay", start=START)
    out = replay.detect(attacked, clean_z(target.index))
    last = labels["end"]
    assert out.loc[last, "exact"] and out.loc[last, "alarm"]
    assert not out.loc[: last - DAY, "alarm"].any()
    assert out.attrs["match"] == "exact"
    assert out.attrs["est_start"] == labels["start"]
    assert out.attrs["source_start"] == labels["source_start"]


def test_noisy_copy_with_context_shift_is_detected(target):
    attacked, labels, z = shifted_replay(target)
    out = replay.detect(attacked, z)
    assert not out["exact"].any()
    assert out.loc[labels["start"]: labels["end"] + 3 * DAY, "alarm"].any()
    assert not out.loc[: labels["start"] - DAY, "alarm"].any()
    assert out.attrs["match"] == "near"
    assert out.attrs["est_start"] == labels["start"]
    assert out.attrs["source_start"] == labels["source_start"]


def test_noisy_copy_without_disagreement_does_not_alarm(target):
    attacked, labels = inject(target, "replay", start=START, noise=0.01, seed=3)
    out = replay.detect(attacked, clean_z(target.index, 1))
    assert not out.loc[labels["start"]: labels["end"], "alarm"].any()


def test_clean_synthetic_has_no_replay_alarms(target):
    out = replay.detect(target, clean_z(target.index, 5))
    assert not out["alarm"].any()


def test_all_zero_windows_are_not_replay(target):
    x = target.copy()
    x.iloc[400:407] = 0.0
    x.iloc[800:807] = 0.0
    out = replay.detect(x, clean_z(x.index), {"stuck_run": 100})
    assert not out["exact"].any()
    assert not out["alarm"].any()


def test_stuck_run_is_flagged(target):
    attacked, labels = inject(target, "stuck", start=START)
    out = replay.detect(attacked, clean_z(target.index))
    s = pd.Timestamp(START)
    assert not out.loc[s: s + 3 * DAY, "stuck"].any()
    assert out.loc[s + 4 * DAY: labels["end"], "stuck"].all()
    assert out.loc[s + 4 * DAY, "alarm"]
    assert out.attrs["match"] == "stuck"
    assert out.attrs["est_start"] == s and out.attrs["source_start"] is None


def test_causality_prefix(target):
    attacked, labels, z = shifted_replay(target)
    full = replay.detect(attacked, z)
    for t in (300, 2250, 2256, 2300):
        part = replay.detect(attacked.iloc[:t], z.iloc[:t])
        pd.testing.assert_frame_equal(part, full.iloc[:t])


def test_nan_days_never_alarm(target):
    x = target.copy()
    x.iloc[500:520] = np.nan
    out = replay.detect(x, clean_z(x.index))
    assert (out["score"].iloc[500:520] == 0).all()
    empty = pd.Series(np.nan, index=target.index[:50])
    assert not replay.detect(empty, None)["alarm"].any()


def test_works_without_z_and_on_short_series(target):
    attacked, labels = inject(target, "replay", start=START)
    assert replay.detect(attacked, None).loc[labels["end"], "exact"]
    short = target.iloc[:5]
    out = replay.detect(short, None)
    assert len(out) == 5 and not out["alarm"].any()


def test_speed_seven_years(target):
    z = clean_z(target.index)
    best = min(_timed(lambda: replay.detect(target, z)) for _ in range(3))
    assert best < 1.0


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def test_params():
    p = replay.merge_params({"m": None, "near_thr": 0.9})
    assert p["m"] == 7 and p["near_thr"] == 0.9
    with pytest.raises(ValueError):
        replay.merge_params({"m": 1})
    with pytest.raises(ValueError):
        replay.merge_params({"near_thr": 1.5})