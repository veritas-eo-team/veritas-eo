"""Tests for app/ui.py helpers (offline, synthetic data; no Streamlit server needed)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.synthetic import make_synthetic_frame
from src.trust.provenance import append_record, load_public_keys, read_log

ROOT = Path(__file__).resolve().parents[1]
CFG = {"test_year": 2024, "eval": {"attack_window": ["02-01", "09-30"], "warmup_days": 365}}


@pytest.fixture(scope="module")
def ui():
    spec = importlib.util.spec_from_file_location("app_ui", ROOT / "app" / "ui.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_result(n: int = 120) -> pd.DataFrame:
    index = pd.date_range("2024-03-01", periods=n, freq="D", name="date")
    action = np.array(["accept"] * n, dtype=object)
    action[40:60] = "quarantine"
    return pd.DataFrame({
        "target": 5.0, "expected": 5.0, "sigma": 0.05, "gap": 0.0, "z": 0.0,
        "step_score": 0.2, "drift_score": 0.1, "replay_score": 0.0, "consistency_score": 0.0,
        "p_compromise": 0.05, "trust": np.where(action == "quarantine", 0.1, 0.95),
        "attack_type": np.where(action == "quarantine", "step", "none"),
        "est_start": pd.NaT, "action": action, "alert": np.arange(n) == 40,
    }, index=index)


def test_attack_and_window(ui):
    df = make_synthetic_frame(seed=1)
    run_df, labels = ui.attack_and_window(df, CFG, "step", 0.12, None, 0.0, "2024-05-01", 1)
    assert labels["start"] == pd.Timestamp("2024-05-01")
    assert run_df.index[0] == pd.Timestamp("2023-05-02") and run_df.index[-1] == pd.Timestamp("2024-12-31")
    ratio = run_df.loc["2024-05-01":, "target"] / df.loc["2024-05-01":, "target"]
    assert np.allclose(ratio, 1.12)
    clean_df, clean_labels = ui.attack_and_window(df, CFG, "none", None, None, 0.0, None, 1)
    assert clean_labels["start"] is None and clean_df.index[0] == pd.Timestamp("2023-01-01")


def test_summarize_run(ui):
    result = fake_result()
    s = ui.summarize_run(result, {"start": pd.Timestamp("2024-04-05"), "end": pd.Timestamp("2024-06-28")})
    assert s["first_alarm"] == pd.Timestamp("2024-04-10") and s["delay"] == 5
    assert s["type"] == "step" and s["lowest_trust"] == pytest.approx(0.1)
    clean = ui.summarize_run(result, {"start": None, "end": None})
    assert clean["first_alarm"] is None and clean["alarm_days"] == 20


def test_plot_and_badge(ui):
    result = fake_result()
    run_df = pd.DataFrame({"ctx_CLRSKY_SFC_SW_DWN": 7.0}, index=result.index)
    labels = {"start": pd.Timestamp("2024-04-05"), "end": pd.Timestamp("2024-06-28")}
    fig = ui.plot_run(result, run_df, labels, upto=pd.Timestamp("2024-05-01"),
                      xlim=(result.index[0], result.index[-1]))
    assert len(fig.axes) == 4
    html = ui.badge_html("quarantine", pd.Timestamp("2024-04-10"), 0.1)
    assert "QUARANTINE" in html and "2024-04-10" in html


def test_record_checks(ui, tmp_path):
    log, keys_dir = tmp_path / "log.jsonl", tmp_path / "keys"
    for i in range(3):
        f = tmp_path / f"f{i}.txt"
        f.write_text(str(i), encoding="utf-8")
        append_record(f, "download", params={"i": i}, log_path=log, keys_dir=keys_dir, root=tmp_path)
    records = read_log(log)
    keys = load_public_keys(keys_dir)
    assert ui.record_checks(records, keys) == [True, True, True]
    records[1]["params"] = {"i": 99}
    assert ui.record_checks(records, keys) == [True, False, True]


def test_altair_chart_and_explanation(ui):
    result = fake_result()
    result.loc[result.index[45], "target"] = 8.0          # above the clear-sky limit below
    result.loc[result.index[45], "step_score"] = 1.4
    run_df = pd.DataFrame({"ctx_CLRSKY_SFC_SW_DWN": 7.0, "nb_0": 5.1, "nb_1": 4.9, "cloud": np.nan},
                          index=result.index)
    labels = {"start": pd.Timestamp("2024-04-05"), "end": pd.Timestamp("2024-06-28")}
    spec = ui.altair_run(result, run_df, labels, xlim=(result.index[0], result.index[-1])).to_dict()
    assert "vconcat" in spec and len(spec["vconcat"]) == 4
    ex = ui.explain_day(result, run_df, result.index[45], {"fusion": {"slope": 8.0}})
    texts = " ".join(t for _, t in ex["reasons"])
    assert "clear-sky limit" in texts and "Step (CUSUM)" in texts
    assert ex["reasons"][0][0] == "strong"
    assert len(ex["contrib"]) == 4 and ex["contrib"]["p"].between(0, 1).all()
    assert len(ui.figure_png(ui.plot_run(result, run_df, labels))) > 1000