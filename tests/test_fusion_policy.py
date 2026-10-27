"""Tests for src/detect/fusion.py and src/respond/policy.py (offline, synthetic data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.detect.fusion as fusion
import src.respond.policy as policy

CFG = {
    "fusion": {"mode": "rules", "slope": 4.0},
    "policy": {"accept": 0.8, "deweight": 0.5, "verify": 0.2, "hold_days": 3},
}


def make_det(n: int = 500, start: str = "2022-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=n, freq="D", name="date")
    return pd.DataFrame({
        "step_score": 0.0, "drift_score": 0.0, "replay_score": 0.0, "consistency_score": 0.0,
        "replay_exact": False, "stuck": False, "gap": 0.0,
        "step_est_start": pd.NaT, "drift_est_start": pd.NaT,
    }, index=index)


def set_from(det: pd.DataFrame, col: str, at: int, value) -> None:
    det.iloc[at:, det.columns.get_loc(col)] = value


def fused_from_p(p, forced=None) -> pd.DataFrame:
    index = pd.date_range("2022-01-01", periods=len(p), freq="D", name="date")
    return pd.DataFrame({"p_compromise": p, "attack_type": "none",
                         "forced": forced if forced is not None else False}, index=index)


# ---------------------------------------------------------------- fusion: probability

def test_rules_formula():
    det = make_det(10)
    det["step_score"] = 1.0
    p = fusion.fuse(det, CFG)["p_compromise"].iloc[0]
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    assert p == pytest.approx(1.0 - 0.5 * (1.0 - sig(-4.0)) ** 3)


def test_p_in_unit_interval_and_nan_scores_count_as_zero():
    rng = np.random.default_rng(0)
    det = make_det(300)
    for col in fusion.SCORE_COLUMNS:
        det[col] = rng.uniform(0.0, 5.0, 300)
    det.iloc[::7, det.columns.get_loc("step_score")] = np.nan
    p = fusion.fuse(det, CFG)["p_compromise"]
    assert p.between(0.0, 1.0).all()
    quiet = make_det(5)
    quiet["drift_score"] = np.nan
    pd.testing.assert_series_equal(fusion.fuse(quiet, CFG)["p_compromise"],
                                   fusion.fuse(make_det(5), CFG)["p_compromise"])


def test_exact_replay_and_stuck_force_high_p():
    det = make_det(300)
    det.iloc[200, det.columns.get_loc("replay_exact")] = True
    det.iloc[100:104, det.columns.get_loc("stuck")] = True
    fused = fusion.fuse(det, CFG)
    assert fused["p_compromise"].iloc[200] >= 0.99 and fused["forced"].iloc[200]
    assert fused["attack_type"].iloc[200] == "replay"
    assert (fused["attack_type"].iloc[100:104] == "stuck").all()
    assert (fused["p_compromise"].iloc[100:104] >= 0.99).all()
    assert fused["attack_type"].iloc[50] == "none"


def test_replay_score_alarm_is_labelled_replay_but_not_forced():
    det = make_det(100)
    det.iloc[60, det.columns.get_loc("replay_score")] = 1.2
    fused = fusion.fuse(det, CFG)
    assert fused["attack_type"].iloc[60] == "replay"
    assert not fused["forced"].iloc[60]


# ---------------------------------------------------------------- fusion: attack type

def test_step_label_uses_step_est_start():
    det = make_det(500)
    gap = np.random.default_rng(1).normal(0.0, 0.05, 500)
    gap[300:] += 0.113
    det["gap"] = gap
    set_from(det, "step_score", 310, 1.5)
    set_from(det, "step_est_start", 310, det.index[300])
    labels = fusion.fuse(det, CFG)["attack_type"]
    assert (labels.iloc[:310] == "none").all()
    assert (labels.iloc[310:] == "step").mean() > 0.9


def test_drift_label_uses_drift_est_start():
    det = make_det(500)
    gap = np.random.default_rng(2).normal(0.0, 0.05, 500)
    k = np.arange(200)
    gap[300:] += 0.14 * np.minimum((k + 1.0) / 90.0, 1.0)
    det["gap"] = gap
    set_from(det, "drift_score", 340, 1.5)
    set_from(det, "drift_est_start", 340, det.index[300])
    labels = fusion.fuse(det, CFG)["attack_type"]
    assert (labels.iloc[:340] == "none").all()
    assert (labels.iloc[340:] == "drift").mean() > 0.8


def test_consistency_only_alarm_still_gets_a_shape_label():
    det = make_det(500)
    gap = np.random.default_rng(3).normal(0.0, 0.05, 500)
    gap[300:] += 0.113
    det["gap"] = gap
    set_from(det, "consistency_score", 305, 1.0)
    labels = fusion.fuse(det, CFG)["attack_type"]
    assert labels.iloc[305:].isin(["step", "drift"]).all()
    assert (labels.iloc[305:] == "step").mean() > 0.8
    assert (labels.iloc[:305] == "none").all()


# ---------------------------------------------------------------- fusion: LR

def lr_data():
    rng = np.random.default_rng(4)
    det = make_det(600)
    y = np.zeros(600, dtype=int)
    y[400:] = 1
    for col in fusion.SCORE_COLUMNS:
        det[col] = np.abs(rng.normal(0.0, 0.3, 600))
    det.iloc[400:, det.columns.get_loc("step_score")] = 1.5 + rng.normal(0.0, 0.3, 200)
    return det, y


def test_lr_round_trip_and_fuse(tmp_path):
    det, y = lr_data()
    model = fusion.fit_fusion(fusion.fusion_features(det), y)
    path = fusion.save_fusion(model, tmp_path / "fusion.joblib")
    loaded = fusion.load_fusion(path)
    X = fusion.fusion_features(det).to_numpy()
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))
    p = fusion.fuse(det, CFG, lr_model=loaded)["p_compromise"]
    assert p.between(0.0, 1.0).all()
    assert p.iloc[400:].mean() > p.iloc[:400].mean() + 0.5


def test_lr_needs_both_classes():
    det, _ = lr_data()
    with pytest.raises(ValueError):
        fusion.fit_fusion(fusion.fusion_features(det), np.zeros(len(det)))


def test_lr_mode_without_model_falls_back_to_rules(monkeypatch):
    def missing(path=None):
        raise FileNotFoundError("no model")
    monkeypatch.setattr(fusion, "load_fusion", missing)
    det, _ = lr_data()
    lr_cfg = {**CFG, "fusion": {"mode": "lr", "slope": 4.0}}
    pd.testing.assert_frame_equal(fusion.fuse(det, lr_cfg), fusion.fuse(det, CFG))


# ---------------------------------------------------------------- policy

def test_thresholds_map_to_actions():
    out = policy.decide(fused_from_p([0.1, 0.35, 0.65]), CFG)
    assert list(out["action"]) == ["accept", "deweight", "verify"]
    assert np.allclose(out["trust"], [0.9, 0.65, 0.35])
    assert not out["alert"].any()
    last = policy.decide(fused_from_p([0.1, 0.9]), CFG)
    assert list(last["action"]) == ["accept", "quarantine"] and last["alert"].iloc[1]


def test_forced_rows_are_quarantined():
    out = policy.decide(fused_from_p([0.0, 0.0, 0.0], forced=[False, True, False]), CFG)
    assert out["action"].iloc[1] == "quarantine" and out["alert"].iloc[1]
    assert out["action"].iloc[2] == "quarantine"  # hysteresis keeps it quarantined


def test_hysteresis_prevents_flicker():
    p = [0.1] * 3 + [0.95] + [0.1] + [0.95] + [0.1] * 5
    out = policy.decide(fused_from_p(p), CFG)
    expected = ["accept"] * 3 + ["quarantine"] * 5 + ["accept"] * 3
    assert list(out["action"]) == expected
    assert list(np.flatnonzero(out["alert"].to_numpy())) == [3]


def test_middle_trust_resets_the_release_count():
    p = [0.95, 0.1, 0.1, 0.35, 0.1, 0.1, 0.1]
    out = policy.decide(fused_from_p(p), CFG)
    assert list(out["action"]) == ["quarantine"] * 6 + ["accept"]


def test_alert_on_each_new_episode():
    p = [0.95] + [0.1] * 3 + [0.95] + [0.1] * 3
    out = policy.decide(fused_from_p(p), CFG)
    assert list(np.flatnonzero(out["alert"].to_numpy())) == [0, 4]


def test_policy_params_validation():
    with pytest.raises(ValueError):
        policy.merge_params({"accept": 0.5, "deweight": 0.6})
    with pytest.raises(ValueError):
        policy.merge_params({"hold_days": 0})
    assert policy.merge_params({"accept": None})["accept"] == 0.8


def test_fuse_then_decide_end_to_end():
    det = make_det(300)
    det.iloc[150, det.columns.get_loc("replay_exact")] = True
    out = policy.decide(fusion.fuse(det, CFG), CFG)
    assert out["action"].iloc[149] == "accept"
    assert out["action"].iloc[150] == "quarantine" and out["alert"].iloc[150]
    assert out["action"].iloc[153] == "accept"  # released after 3 good days



def test_shape_params_from_config():
    det = make_det(500)
    gap = np.random.default_rng(2).normal(0.0, 0.05, 500)
    k = np.arange(200)
    gap[300:] += 0.14 * np.minimum((k + 1.0) / 90.0, 1.0)
    det["gap"] = gap
    set_from(det, "drift_score", 340, 1.5)
    strict = {**CFG, "fusion": {"mode": "rules", "slope": 8.0, "shape_penalty": 0.99}}
    labels = fusion.fuse(det, strict)["attack_type"]
    assert not (labels == "drift").any()            # a huge penalty never picks a ramp
    assert fusion.shape_params({"shape_min_ramp": 10})["min_ramp_days"] == 10
    assert fusion.shape_params(None)["lookback"] == fusion.SHAPE_LOOKBACK