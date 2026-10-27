"""Tests for src/eval (metrics on tiny hand-made examples; quick benchmark on synthetic data)."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

import src.eval.metrics as metrics
import src.eval.run_benchmark as rb
import src.eval.scenarios as scenarios
from src.detect.context_model import ContextModel
from src.synthetic import make_synthetic_frame

CONTEXT_VARS = ["CLRSKY_SFC_SW_DWN", "T2M", "RH2M", "PRECTOTCORR", "GWETTOP"]
CFG = {
    "test_year": 2024,
    "eval": {"n_seeds": 2, "attack_window": ["02-01", "09-30"], "warmup_days": 365},
    "fusion": {"mode": "rules", "slope": 8.0},
    "policy": {"accept": 0.8, "deweight": 0.5, "verify": 0.2, "hold_days": 3},
    "step": {"phi": 0.3, "k": 0.75, "h": 8.0}, "drift": {"L": 4.0},
    "replay": {}, "consistency": {}, "context_model": {},
}


def c5_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", "2024-12-31", freq="D", name="date")
    return pd.DataFrame({
        "target": 5.0, "expected": 5.0, "sigma": 0.05, "gap": 0.0, "z": 0.0,
        "step_score": 0.0, "drift_score": 0.0, "replay_score": 0.0, "consistency_score": 0.0,
        "p_compromise": 0.01, "trust": 0.99, "attack_type": "none", "est_start": pd.NaT,
        "action": "accept", "alert": False,
    }, index=index)


def put(df: pd.DataFrame, col: str, lo: int, hi: int, value) -> None:
    df.iloc[lo:hi, df.columns.get_loc(col)] = value


def labels_for(df: pd.DataFrame, start: int, end: int, kind: str = "step") -> dict:
    return {"type": kind, "start": df.index[start], "end": df.index[end], "size": 0.12, "params": {}}


# ---------------------------------------------------------------- event metrics

def test_event_metrics_hand_made():
    df = c5_frame()
    put(df, "action", 100, 101, "verify")                  # false alarm
    put(df, "action", 200, 211, "quarantine")              # attack alarm
    put(df, "attack_type", 200, 211, "step")
    put(df, "est_start", 200, 211, df.index[196])
    put(df, "step_score", 199, 211, 1.5)
    put(df, "drift_score", 202, 211, 1.2)
    put(df, "trust", 195, 206, 0.1)
    put(df, "action", 300, 302, "verify")                  # false alarm after the grace period
    ev = metrics.event_metrics(df, labels_for(df, 195, 205), 2024)
    assert ev["caught"] is True and ev["delay_days"] == 5
    assert ev["fa_episodes"] == 2
    assert ev["fa_days"] == 366 - 11 - 30
    assert ev["type_correct"] is True and ev["start_error_days"] == 1
    assert ev["first_detector"] == "step"
    assert ev["min_trust_attacked"] == pytest.approx(0.1)


def test_grace_period_and_misses():
    df = c5_frame()
    put(df, "action", 220, 222, "verify")                  # inside end + 30: caught, not a false alarm
    ev = metrics.event_metrics(df, labels_for(df, 195, 205), 2024)
    assert ev["caught"] is True and ev["delay_days"] == 25 and ev["fa_episodes"] == 0
    assert ev["first_detector"] == "fusion"
    late = c5_frame()
    put(late, "action", 250, 252, "verify")                # after the grace period: missed + false alarm
    ev2 = metrics.event_metrics(late, labels_for(late, 195, 205), 2024)
    assert ev2["caught"] is False and np.isnan(ev2["delay_days"]) and ev2["fa_episodes"] == 1


def test_clean_run_counts_false_alarms():
    df = c5_frame()
    put(df, "action", 10, 12, "verify")
    put(df, "action", 40, 41, "quarantine")
    ev = metrics.event_metrics(df, {"type": "none", "start": None, "end": None, "size": None, "params": {}}, 2024)
    assert ev["fa_episodes"] == 2 and ev["fa_days"] == 366 and np.isnan(ev["caught"])


def test_fa_per_location_year():
    assert metrics.fa_per_location_year(2, 730.5) == pytest.approx(1.0)
    assert np.isnan(metrics.fa_per_location_year(1, 0))


# ---------------------------------------------------------------- per-day and calibration

def test_per_day_metrics():
    m = metrics.per_day_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], [0, 1, 1, 0])
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["roc_auc"] == pytest.approx(0.75)
    assert np.isnan(metrics.per_day_metrics([0, 0], [0.1, 0.2], [0, 1])["roc_auc"])


def test_calibration():
    brier, table = metrics.calibration([0, 1, 1, 1], [0.05, 0.05, 0.95, 0.95])
    assert brier == pytest.approx(0.2275)
    assert len(table) == 10 and table["n"].sum() == 4
    assert table.loc[0, "frac_positive"] == 0.5 and table.loc[9, "frac_positive"] == 1.0


# ---------------------------------------------------------------- scenarios

def test_make_scenarios_counts_and_windows():
    scns = scenarios.make_scenarios("test", n_seeds=3, cfg=CFG, locations=["a", "b"])
    assert len(scns) == 2 * (1 + 3 * 3)
    assert {s["window_start"] for s in scns} == {"2024-02-01"}
    assert {s["window_end"] for s in scns} == {"2024-09-30"}
    assert len({s["id"] for s in scns}) == len(scns)
    json.dumps(scns)
    quick = scenarios.make_scenarios("test", n_seeds=3, cfg=CFG, locations=["a", "b"], sweep=True, quick=True)
    assert len(quick) == len(scns) + 2 * 5
    full = scenarios.make_scenarios("test", n_seeds=3, cfg=CFG, locations=["a", "b"], sweep=True)
    assert len(full) == len(scns) + 2 * (11 + 20 + 9)


def test_adaptive_scenarios_and_measured_limit():
    adapt = scenarios.adaptive_scenarios(0.06, ["a"], 2, CFG)
    assert len(adapt) == 2 and {s["family"] for s in adapt} == {"adaptive"}
    assert adapt[0]["size"] == pytest.approx(0.054) and adapt[0]["adaptive_limit"] == 0.06
    records = pd.DataFrame({
        "family": "sweep_step",
        "size": [0.03, 0.03, 0.05, 0.05, 0.08, 0.08, -0.05],
        "caught": [False, False, True, False, True, True, True],
    })
    assert metrics.measured_limit(records) == 0.05
    assert metrics.measured_limit(records[records["size"] == 0.03]) is None


def test_system_info_is_anonymous():
    info = metrics.system_info()
    for key in ("os", "python", "cores_logical", "ram_gb"):
        assert key in info
    assert not {"hostname", "node", "user"} & set(info)
    json.dumps(info)


# ---------------------------------------------------------------- quick benchmark

@pytest.fixture(scope="module")
def model():
    frames = [make_synthetic_frame(location=f"train{i}", seed=i) for i in range(3)]
    return ContextModel(alpha=1.0, sigma_min=0.02, test_year=2024, context_vars=CONTEXT_VARS).fit(frames)


def test_quick_benchmark_on_synthetic(model, tmp_path):
    test_frames = {"syn_test": make_synthetic_frame(location="syn_test", seed=11, split="test")}
    train_frames = {"train0": make_synthetic_frame(location="train0", seed=0)}
    t0 = time.perf_counter()
    res = rb.run(CFG, model, test_frames, train_frames, n_seeds=1, sweep=True, quick=True, out_dir=tmp_path)
    assert time.perf_counter() - t0 < 30
    for path in res["paths"].values():
        assert path.exists() and path.stat().st_size > 0
    table = res["table"]
    assert set(table["group"]) >= {"test", "unseen_year"}
    assert set(table.loc[table["group"] == "test", "attack"]) >= {"none", "step", "drift", "replay"}
    assert table.loc[table["attack"] == "none", "fa_per_location_year"].notna().all()
    summary = res["paths"]["summary.md"].read_text(encoding="utf-8")
    assert "Standard scenario" in summary and "Unseen year" in summary