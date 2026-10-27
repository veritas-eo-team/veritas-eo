"""Tests for src/eval/tune.py (offline, synthetic train frames)."""
from __future__ import annotations

import time

import pandas as pd
import pytest
import yaml

import src.common as common
import src.eval.tune as tune
from src.synthetic import make_synthetic_frame

CFG = {
    "test_year": 2024,
    "context_vars": ["CLRSKY_SFC_SW_DWN", "T2M", "RH2M", "PRECTOTCORR", "GWETTOP"],
    "context_model": {},
    "eval": {"attack_window": ["02-01", "09-30"], "warmup_days": 365},
    "fusion": {"mode": "rules", "slope": 8.0},
    "policy": {"accept": 0.8, "deweight": 0.5, "verify": 0.2, "hold_days": 3},
    "step": {"phi": 0.3, "k": 0.75, "h": 8.0}, "drift": {"L": 4.0},
    "replay": {}, "consistency": {},
}


@pytest.fixture(scope="module")
def frames():
    return [make_synthetic_frame(location=f"train{i}", seed=i) for i in range(3)]


def test_refuses_test_split(frames):
    bad = frames[:2] + [make_synthetic_frame(location="hidden", seed=9, split="test")]
    with pytest.raises(ValueError):
        tune.tune(bad, CFG, quick=True, write=False)


def test_choose_prefers_catch_within_budget():
    cands = [
        {"value": 4, "fa": 2.0, "catch": 1.0, "delay": 2.0},
        {"value": 8, "fa": 0.25, "catch": 0.9, "delay": 5.0},
        {"value": 12, "fa": 0.1, "catch": 0.9, "delay": 9.0},
        {"value": 20, "fa": 0.0, "catch": 0.5, "delay": 30.0},
    ]
    assert tune.choose(cands, 0.3)["value"] == 12      # same catch, fewer false alarms
    assert tune.choose(cands, 0.05)["value"] == 20
    assert tune.choose(cands, -1.0)["value"] == 20     # nothing fits: fewest false alarms


def test_attacked_series_shift_gap_and_use_years_before_test_year(frames):
    clean = tune.build_clean(frames, CFG)
    assert all(s.df.index.max().year < 2024 for s in clean)
    attacked = tune.build_attacked(clean[:1], tune.eval_settings(CFG), seeds=1, years=[2023], attacks=("step",))
    s, a = clean[0], attacked[0]
    shift = (a.gap - s.gap)[a.attacked]
    assert shift.mean() == pytest.approx(0.113, abs=0.003)
    pd.testing.assert_series_equal(a.sigma, s.sigma)
    assert (a.gap - s.gap)[~a.attacked].abs().max() == 0


def test_tune_quick_writes_valid_yaml(frames, tmp_path):
    out = tmp_path / "tuned.yaml"
    t0 = time.perf_counter()
    res = tune.tune(frames, CFG, quick=True, out_path=out, models_dir=tmp_path)
    assert time.perf_counter() - t0 < 30
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    for section in ("step", "drift", "replay", "consistency", "tuning"):
        assert section in data
    assert 0.0 <= data["step"]["phi"] < 0.95 and data["step"]["k"] > 0
    assert data["step"]["h"] in tune.QUICK_GRIDS["step_h"]
    assert data["consistency"]["min_violations"] in tune.QUICK_GRIDS["cons_minv"]
    assert (tmp_path / "fusion.joblib").exists()
    merged = common.load_config(tuned_path=out)     # the real settings.yaml + this tuned.yaml
    assert merged["step"]["h"] == data["step"]["h"]
    assert {"before", "after"} <= set(res["table"].columns)
    assert "system FA/location-year" in set(res["table"]["metric"])