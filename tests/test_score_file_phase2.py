"""score_file Phase 2: no truth download, history before the file, no-coordinate mode, sub-daily input."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import score_file as sf
from src.common import load_config
from src.detect.context_model import ContextModel
from src.pipeline import ALARM_ACTIONS, C5_COLUMNS
from src.synthetic import make_synthetic_frame

CELLS = ("nb_N", "nb_E", "nb_S", "nb_W")


@pytest.fixture(scope="module")
def model_and_cfg():
    cfg = load_config()
    frames = [make_synthetic_frame(location=f"syn{i}", seed=i, split="train") for i in range(1, 5)]
    return ContextModel(cfg=cfg).fit(frames), cfg


def _fake_context(frame: pd.DataFrame):
    """fetch_context stand-in built from a synthetic frame; never returns the target."""
    def fake(lat, lon, start, end, cfg, cache_dir=None):
        sub = frame.loc[pd.Timestamp(start):pd.Timestamp(end)]
        site = pd.DataFrame({c[4:]: sub[c] for c in sub.columns if c.startswith("ctx_")}, index=sub.index)
        nbs = sorted(c for c in sub.columns if c.startswith("nb_"))
        return site, {CELLS[i]: sub[c] for i, c in enumerate(nbs)}, {}
    return fake


def test_site_request_never_includes_target(monkeypatch, model_and_cfg):
    _, cfg = model_and_cfg
    tv, seen = cfg["target"], []

    def fake_cell(lat, lon, params, start, end, session, cache):
        seen.append(list(params))
        idx = pd.date_range(start, end, freq="D", name="date")
        cols = list(dict.fromkeys(list(params) + [tv]))  # simulate an old cache that still holds the target
        return pd.DataFrame(1.0, index=idx, columns=cols), {}

    monkeypatch.setattr(sf, "_fetch_cell", fake_cell)
    site, cands, _ = sf.fetch_context(36.8, 10.18, "2024-01-01", "2024-01-31", cfg)
    ctx = [v for v in cfg["context_vars"] if v != tv]
    site_calls = [p for p in seen if any(v in p for v in ctx)]
    assert site_calls and all(tv not in p for p in site_calls)
    assert tv not in site.columns
    assert len(cands) == 4


def test_history_guard_rejects_overlap():
    start = pd.Timestamp("2024-01-01")
    with pytest.raises(sf.OracleGuardError):
        sf._check_history(pd.date_range("2023-01-01", "2024-01-05", freq="D"), start)
    sf._check_history(pd.date_range("2023-01-01", "2023-12-31", freq="D"), start)


def test_same_cell_neighbour_dropped_even_when_file_is_attacked():
    idx = pd.date_range("2023-01-01", periods=200, freq="D", name="date")
    rng = np.random.default_rng(0)
    clean = pd.Series(15 + rng.normal(0, 3, 200), index=idx).round(2)
    attacked = clean.copy()
    attacked.iloc[120:] *= 1.12  # 40% of the file attacked
    other = (clean * 0.9 + rng.normal(0, 1, 200)).round(2)
    kept, dropped = sf.pick_neighbours(attacked, {"nb_N": clean, "nb_E": other})
    assert "nb_N" in dropped and "nb_E" in kept


def test_short_file_uses_history_before_file_and_catches_step(monkeypatch, model_and_cfg):
    model, cfg = model_and_cfg
    frame = make_synthetic_frame(location="victim", seed=11, split="test")
    calls = {}

    def fake_history(lat, lon, start, end, cfg_, cache_dir=None):
        calls["range"] = (pd.Timestamp(start), pd.Timestamp(end))
        return frame["target"].loc[pd.Timestamp(start):pd.Timestamp(end)]

    monkeypatch.setattr(sf, "fetch_history", fake_history)
    monkeypatch.setattr(sf, "fetch_context", _fake_context(frame))
    data = frame["target"].loc["2024-01-01":"2024-12-31"].copy()
    start = pd.Timestamp("2024-03-01")
    data.loc[start:] *= 1.12

    result, info = sf.score_series(data, cfg, model, lat=10.0, lon=10.0, cache=False)
    assert info["history"] == "nasa" and not info["reference_in_file"]
    assert len(info["neighbours"]) >= 2
    assert calls["range"][1] < data.index[0]
    assert list(result.columns[:len(C5_COLUMNS)]) == C5_COLUMNS and "reasons" in result
    assert result.index[0] == data.index[0] and len(result) == len(data)
    alarm = result["action"].isin(ALARM_ACTIONS)
    assert alarm.loc[start:start + pd.Timedelta(days=30)].any()
    assert alarm.loc[:start - pd.Timedelta(days=1)].mean() < 0.1


def test_no_coordinates_runs_replay_checks_only(model_and_cfg):
    _, cfg = model_and_cfg
    frame = make_synthetic_frame(location="nocoord", seed=5, start="2022-01-01", end="2023-12-31")
    t = frame["target"].copy()
    t.iloc[500:507] = t.iloc[100:107].to_numpy()
    with pytest.raises(sf.InputError):
        sf.score_series(t, cfg)
    result, info = sf.score_series(t, cfg, allow_no_context=True)
    assert info["history"] == "none" and info["notes"]
    assert list(result.columns[:len(C5_COLUMNS)]) == C5_COLUMNS
    assert (result["step_score"] == 0).all()
    assert (result["action"].iloc[506:512] == "quarantine").any()


def _write_hourly(tmp_path, name, column, value, drop=None):
    stamps = pd.date_range("2024-01-01", periods=24 * 70, freq="h")
    df = pd.DataFrame({"timestamp": stamps, column: value})
    if drop is not None:
        df = df.drop(index=drop)
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_hourly_wm2_is_averaged_and_partial_days_dropped(tmp_path):
    path = _write_hourly(tmp_path, "hourly.csv", "ghi", 200.0, drop=range(24 * 5, 24 * 5 + 10))
    target, _ = sf.read_input(path, units="wm2")
    assert len(target) == 70
    assert target.iloc[0] == pytest.approx(200 * 0.0864)
    assert np.isnan(target.iloc[5])  # 14 of 24 hours, under 80%
    assert "daily mean" in target.attrs["aggregation"]
    with pytest.raises(sf.InputError):
        sf.read_input(path, units="wm2", agg="sum")


def test_hourly_energy_is_summed(tmp_path):
    path = _write_hourly(tmp_path, "energy.csv", "value", 0.25)  # kWh per hour
    target, _ = sf.read_input(path, units="kwh")
    assert target.iloc[0] == pytest.approx(6.0 * 3.6)