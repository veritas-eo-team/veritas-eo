"""Tests for src/pipeline.py (offline, synthetic data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.common as common
import src.pipeline as pipeline
from src.attacks.inject import inject
from src.detect.context_model import ContextModel
from src.synthetic import make_synthetic_frame

CONTEXT_VARS = ["CLRSKY_SFC_SW_DWN", "T2M", "RH2M", "PRECTOTCORR", "GWETTOP"]
CFG = {
    "test_year": 2024,
    "fusion": {"mode": "rules", "slope": 8.0},
    "policy": {"accept": 0.8, "deweight": 0.5, "verify": 0.2, "hold_days": 3},
    "step": {"phi": 0.3, "k": 0.75, "h": 8.0}, "drift": {"L": 4.0},
    "replay": {}, "consistency": {}, "context_model": {},
}
ATTACK_START = "2024-03-01"   # position 790 in a frame starting 2022-01-01
SAT_COLUMNS = ["ndvi", "cloud", "days_since_clear", "ctx_CLRSKY_SFC_SW_DWN"]


@pytest.fixture(scope="module")
def model():
    frames = [make_synthetic_frame(location=f"train{i}", seed=i) for i in range(3)]
    return ContextModel(alpha=1.0, sigma_min=0.02, test_year=2024, context_vars=CONTEXT_VARS).fit(frames)


@pytest.fixture(scope="module")
def frame():
    return make_synthetic_frame(location="site", seed=7, start="2022-01-01", end="2024-12-31")


@pytest.fixture(scope="module")
def clean_out(model, frame):
    return pipeline.run_pipeline(frame, model, CFG)


def attacked_frame(frame: pd.DataFrame, attack: str, **kwargs):
    df = frame.copy()
    df["target"], labels = inject(frame["target"], attack, start=ATTACK_START, **kwargs)
    return df, labels


def test_exact_c5_columns(clean_out, frame):
    assert list(clean_out.columns) == pipeline.C5_COLUMNS
    assert clean_out.index.equals(frame.index)
    assert set(clean_out["action"]) <= {"accept", "deweight", "verify", "quarantine"}
    assert set(clean_out["attack_type"]) <= {"none", "step", "drift", "replay", "stuck"}
    assert clean_out["alert"].dtype == bool
    np.testing.assert_allclose(clean_out["trust"], 1.0 - clean_out["p_compromise"])
    np.testing.assert_allclose(clean_out["target"], frame["target"])


def test_clean_series_stays_mostly_accept(clean_out):
    assert (clean_out["action"] == "accept").mean() >= 0.9
    assert clean_out["alert"].sum() <= 2


@pytest.mark.parametrize("attack", ["step", "drift", "replay"])
def test_attacks_lead_to_non_accept_actions(model, frame, attack):
    df, labels = attacked_frame(frame, attack)
    out = pipeline.run_pipeline(df, model, CFG)
    after = out.loc[labels["start"]: labels["end"] + pd.Timedelta(days=60), "action"]
    assert (after != "accept").any()


def test_exact_replay_is_quarantined_with_replay_type(model, frame):
    df, labels = attacked_frame(frame, "replay")
    out = pipeline.run_pipeline(df, model, CFG)
    end = labels["end"]
    assert out.loc[end, "action"] == "quarantine"
    assert out.loc[end, "attack_type"] == "replay"
    assert out.loc[end, "est_start"] == labels["start"]


def test_est_start_filled_from_first_alarm(model, frame):
    df, labels = attacked_frame(frame, "step")
    out = pipeline.run_pipeline(df, model, CFG)
    alarm = out["action"].isin(["verify", "quarantine"])
    first = alarm[alarm].index[0]
    assert out.loc[: first - pd.Timedelta(days=1), "est_start"].isna().all()
    assert out.loc[first:, "est_start"].notna().all()
    after = alarm.loc[labels["start"]:]
    first_after = after[after].index[0]
    assert abs((out.loc[first_after, "est_start"] - labels["start"]).days) <= 30


def test_causality_prefix(model, frame):
    df, _ = attacked_frame(frame, "step")
    full = pipeline.run_pipeline(df, model, CFG)
    for t in (300, 500, 790, 800, 900):
        part = pipeline.run_pipeline(df.iloc[:t], model, CFG)
        pd.testing.assert_frame_equal(part.drop(columns="est_start"),
                                      full.iloc[:t].drop(columns="est_start"))


def test_runs_without_satellite_or_clearsky(model, frame):
    bare = frame.drop(columns=SAT_COLUMNS)
    out = pipeline.run_pipeline(bare, model, CFG)
    assert list(out.columns) == pipeline.C5_COLUMNS
    assert (out["consistency_score"] == 0).all()
    blank = frame.copy()
    blank[SAT_COLUMNS] = np.nan
    assert list(pipeline.run_pipeline(blank, model, CFG).columns) == pipeline.C5_COLUMNS


def test_missing_model_gives_train_hint(frame, tmp_path, monkeypatch):
    monkeypatch.setitem(common.PATHS, "models", tmp_path)
    with pytest.raises(FileNotFoundError, match="src.detect.context_model --train"):
        pipeline.run_pipeline(frame, None, CFG)


def test_timed_run(model, frame, clean_out):
    result, ms_per_day, peak_mb = pipeline.timed_run(frame, model, CFG)
    pd.testing.assert_frame_equal(result, clean_out)
    assert ms_per_day > 0 and peak_mb > 0