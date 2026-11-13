import numpy as np
import pandas as pd
import pytest

from src import score_file as SF
from src.common import load_config
from src.data.align import load


def write_csv(tmp_path, df, name="in.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def days(n, start="2020-01-01"):
    return pd.date_range(start, periods=n).strftime("%Y-%m-%d")


def test_read_input_detects_columns(tmp_path):
    p = write_csv(tmp_path, pd.DataFrame({"Date": days(100), "ghi": np.linspace(3, 6, 100)}))
    s, extras = SF.read_input(p)
    assert len(s) == 100 and s.index.name == "date"
    assert abs(s.iloc[0] - 3.0) < 1e-9
    assert extras.shape[1] == 0


def test_read_input_fills_missing_days(tmp_path):
    keep = np.ones(100, bool)
    keep[10] = False
    p = write_csv(tmp_path, pd.DataFrame({"date": days(100)[keep], "value": 5.0}))
    s, _ = SF.read_input(p)
    assert len(s) == 100 and np.isnan(s.iloc[10])


def test_units_conversion_and_sanity_check(tmp_path):
    p = write_csv(tmp_path, pd.DataFrame({"date": days(100), "value": 200.0}))
    with pytest.raises(SF.InputError, match="too high"):
        SF.read_input(p)
    s, _ = SF.read_input(p, units="wm2")
    assert abs(s.iloc[0] - 17.28) < 1e-9
    k = write_csv(tmp_path, pd.DataFrame({"date": days(100), "value": 5.0}), "k.csv")
    s, _ = SF.read_input(k, units="kwh")
    assert abs(s.iloc[0] - 18.0) < 1e-9
    assert "unit_warning" in SF.read_input(k)[0].attrs


def test_duplicate_days_rejected(tmp_path):
    dates = ["2020-01-01"] + list(days(99))
    p = write_csv(tmp_path, pd.DataFrame({"date": dates, "value": 5.0}))
    with pytest.raises(SF.InputError, match="same day"):
        SF.read_input(p)


def test_needs_location_or_columns():
    s = pd.Series(5.0, index=pd.date_range("2020-01-01", periods=100, name="date"))
    with pytest.raises(SF.InputError, match="lat"):
        SF.score_series(s)


def test_bad_coordinates_rejected():
    s = pd.Series(5.0, index=pd.date_range("2020-01-01", periods=100, name="date"))
    with pytest.raises(SF.InputError, match="lat must be"):
        SF.score_series(s, lat=95.0, lon=10.0)


def _kairouan():
    cfg = load_config()
    return cfg, load("kairouan", cfg).loc["2023-01-01":"2024-12-31"]


def _attacked(df, start="2024-05-01", size=0.12):
    t = df["target"].copy()
    t.loc[start:] *= 1 + size
    return t


def test_end_to_end_with_columns_in_file(tmp_path):
    cfg, df = _kairouan()
    out = df[[c for c in df.columns if c.startswith(("nb_", "ctx_"))]].copy()
    out.insert(0, "target", _attacked(df).to_numpy())
    out.insert(0, "date", df.index.strftime("%Y-%m-%d"))
    s, extras = SF.read_input(write_csv(tmp_path, out))
    result, info = SF.score_series(s, cfg, extras=extras)
    alarm = result["action"].isin(SF.ALARM_ACTIONS)
    assert alarm.loc["2024-05-01":"2024-05-31"].any()
    assert result.loc[alarm, "reasons"].str.len().gt(0).all()
    assert info["mode"] == "columns from file"
    assert SF.alarm_episodes(result)


def test_end_to_end_with_mocked_download(monkeypatch):
    cfg, df = _kairouan()
    tv = cfg["target"]
    site = pd.DataFrame({tv: df["target"], **{c[4:]: df[c] for c in df.columns if c.startswith("ctx_")}})
    cands = {cell: df[f"nb_{i}"] for i, cell in enumerate(["nb_N", "nb_E", "nb_S", "nb_W"]) if f"nb_{i}" in df}
    monkeypatch.setattr(SF, "fetch_context", lambda *a, **k: (site, cands, {tv: "kWh/m^2/day"}))
    result, info = SF.score_series(_attacked(df), cfg, lat=35.68, lon=10.1, cache=False)
    assert result["action"].isin(SF.ALARM_ACTIONS).loc["2024-05-01":"2024-05-31"].any()
    assert info["mode"] == "NASA POWER download"
    assert len(info["neighbours"]) >= 2