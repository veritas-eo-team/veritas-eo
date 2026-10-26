"""Tests for src/data/align.py: fake raw files in a temporary project root, no internet."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import src.common as common
import src.data.align as align
from src.data import power
from src.synthetic import make_synthetic_frame
from src.trust.provenance import file_sha256, read_log
from src.trust.verify import verify_log

TARGET = "ALLSKY_SFC_SW_DWN"
CTX = ["CLRSKY_SFC_SW_DWN", "T2M", "RH2M", "PRECTOTCORR", "GWETTOP"]
UNITS = "kW-hr/m^2/day"
START, END = "2018-01-01", "2018-12-31"

# Sentinel-2 rows: (day offset, ndvi_mean, cloud_fraction, valid_fraction)
S2_ROWS = [
    (10, 0.50, 0.10, 0.90),  # clear
    (15, 0.30, 0.80, 1.00),  # cloudy: cloud kept, NDVI not used
    (20, 0.60, 0.10, 0.30),  # too few valid pixels: ignored
    (25, 0.70, 0.05, 0.95),  # clear
]


def day(i: int) -> pd.Timestamp:
    return pd.Timestamp(START) + pd.Timedelta(days=i)


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def cfg():
    return {
        "target": TARGET,
        "context_vars": list(CTX),
        "start": START,
        "end": END,
        "locations": [
            {"id": "tunis", "lat": 36.80, "lon": 10.18, "climate": "mediterranean", "split": "train"},
            {"id": "kairouan", "lat": 35.68, "lon": 10.10, "climate": "semi_arid", "split": "test"},
        ],
    }


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Point every PATHS entry at a temporary project root."""
    for key, path in common.build_paths(tmp_path).items():
        monkeypatch.setitem(common.PATHS, key, path)
    monkeypatch.delenv("PROVENANCE_KEY_PASSPHRASE", raising=False)
    return tmp_path


# ---------------------------------------------------------------- fake raw files

def power_json(columns: dict[str, pd.Series]) -> dict:
    parameter = {
        var: {d.strftime("%Y%m%d"): (-999.0 if pd.isna(v) else float(v)) for d, v in s.items()}
        for var, s in columns.items()
    }
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [10.18, 36.80, 10.0]},
        "properties": {"parameter": parameter},
        "header": {"fill_value": -999.0},
        "parameters": {var: {"units": UNITS, "longname": var} for var in columns},
        "messages": [],
    }


def write_cell(loc: str, cell_id: str, columns: dict[str, pd.Series]) -> None:
    path = power.cell_path(loc, cell_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(power_json(columns)), encoding="utf-8")


def write_raw(loc: str = "tunis", *, neighbours: bool = True) -> pd.DataFrame:
    """Site + 4 neighbours: nb_E identical to the site, nb_S 40% missing."""
    f = make_synthetic_frame(loc, START, END, seed=1)
    target = f["target"].copy()
    target.iloc[3] = np.nan  # written as -999
    site = {TARGET: target, **{var: f[f"ctx_{var}"] for var in CTX}}
    site = {var: s.drop(day(100)) for var, s in site.items()}  # a date absent from the file
    write_cell(loc, "site", site)
    if neighbours:
        write_cell(loc, "nb_N", {TARGET: f["nb_0"]})
        write_cell(loc, "nb_E", {TARGET: target})
        holey = f["nb_2"].copy()
        holey.iloc[: int(0.4 * len(holey))] = np.nan
        write_cell(loc, "nb_S", {TARGET: holey})
        write_cell(loc, "nb_W", {TARGET: f["nb_3"]})
    return f


def write_s2(loc: str = "tunis", rows=S2_ROWS) -> None:
    lines = ["date,ndvi_mean,cloud_fraction,valid_fraction",
             "2017-12-30,0.9,0.0,1.0"]  # before the config range: ignored
    lines += [f"{day(d):%Y-%m-%d},{n},{c},{v}" for d, n, c, v in rows]
    path = common.PATHS["raw_s2"] / f"{loc}.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def expected_columns(n_nb: int) -> list[str]:
    return (["target"] + [f"nb_{i}" for i in range(n_nb)]
            + [f"ctx_{v}" for v in CTX] + ["ndvi", "cloud", "days_since_clear"])


# ---------------------------------------------------------------- tests

def test_c1_columns_index_and_attrs(repo, cfg):
    write_raw()
    write_s2()
    df = align.build_clean("tunis", cfg)
    assert list(df.columns) == expected_columns(2)
    assert (df.dtypes == "float64").all()
    idx = df.index
    assert idx.name == "date" and idx.tz is None
    assert idx[0] == pd.Timestamp(START) and idx[-1] == pd.Timestamp(END)
    assert len(idx) == 365
    assert bool((idx[1:] - idx[:-1] == pd.Timedelta(days=1)).all())
    assert set(df.attrs) == set(align.ATTR_KEYS)
    assert df.attrs["units"] == UNITS
    assert df.attrs["split"] == "train" and df.attrs["target_var"] == TARGET
    align.validate_c1(df)


def test_fill_values_and_missing_dates_become_nan(repo, cfg):
    write_raw()
    df = align.build_clean("tunis", cfg)
    assert np.isnan(df.loc[day(3), "target"])
    assert day(100) in df.index
    assert np.isnan(df.loc[day(100), "target"])
    assert np.isnan(df.loc[day(100), "ctx_T2M"])
    assert df["target"].notna().sum() == 363


def test_neighbour_selection(repo, cfg):
    f = write_raw()
    df, info = align.build_location("tunis", cfg)
    assert info["neighbours"] == {"nb_0": "nb_N", "nb_1": "nb_W"}
    assert set(info["neighbours_dropped"]) == {"nb_E", "nb_S"}
    assert "identical" in info["neighbours_dropped"]["nb_E"]
    assert "missing" in info["neighbours_dropped"]["nb_S"]
    np.testing.assert_allclose(df["nb_0"].to_numpy(), f["nb_0"].to_numpy())
    np.testing.assert_allclose(df["nb_1"].to_numpy(), f["nb_3"].to_numpy())


def test_no_neighbours_still_c1(repo, cfg):
    write_raw(neighbours=False)
    df, info = align.build_location("tunis", cfg)
    assert list(df.columns) == expected_columns(0)
    assert set(info["neighbours_dropped"].values()) == {"not downloaded"}
    align.validate_c1(df)


def test_s2_ndvi_cloud_and_days_since_clear(repo, cfg):
    write_raw()
    write_s2()
    df, info = align.build_location("tunis", cfg)
    assert df.loc[: day(9), "ndvi"].isna().all()
    assert (df.loc[day(10): day(24), "ndvi"] == 0.5).all()
    assert (df.loc[day(25):, "ndvi"] == 0.7).all()
    assert df.loc[day(10), "cloud"] == pytest.approx(0.1)
    assert df.loc[day(15), "cloud"] == pytest.approx(0.8)
    assert np.isnan(df.loc[day(20), "cloud"])   # valid_fraction too low
    assert np.isnan(df.loc[day(11), "cloud"])   # no acquisition
    assert df["cloud"].notna().sum() == 3
    assert np.isnan(df.loc[day(9), "days_since_clear"])
    assert df.loc[day(10), "days_since_clear"] == 0
    assert df.loc[day(15), "days_since_clear"] == 5
    assert df.loc[day(24), "days_since_clear"] == 14
    assert df.loc[day(25), "days_since_clear"] == 0
    assert df.loc[day(30), "days_since_clear"] == 5
    assert info["clear_images"] == 2
    assert info["s2_acquisitions"] == 4


def test_forward_fill_is_causal(repo, cfg):
    write_raw()
    write_s2()
    full = align.build_clean("tunis", cfg)
    write_s2(rows=[r for r in S2_ROWS if r[0] <= 22])  # future acquisitions removed
    part = align.build_clean("tunis", cfg)
    cut = day(22)
    for col in align.S2_COLUMNS:
        pd.testing.assert_series_equal(full.loc[:cut, col], part.loc[:cut, col])
    assert full.loc[day(24), "ndvi"] == 0.5  # no interpolation towards the next value


def test_load_without_s2_file(repo, cfg):
    write_raw()
    df = align.load("tunis", cfg)  # builds the parquet because it is missing
    assert align.clean_path("tunis").exists()
    assert list(df.columns) == expected_columns(2)
    for col in align.S2_COLUMNS:
        assert df[col].isna().all()
    align.validate_c1(df)


def test_load_round_trip(repo, cfg):
    write_raw()
    write_s2()
    built = align.build_clean("tunis", cfg)
    loaded = align.load("tunis", cfg)
    pd.testing.assert_frame_equal(loaded, built)
    assert loaded.attrs == built.attrs
    assert align.read_info("tunis")["neighbours"] == {"nb_0": "nb_N", "nb_1": "nb_W"}


def test_provenance_record_and_rebuild_rules(repo, cfg):
    write_raw()
    write_s2()
    _, info = align.build_location("tunis", cfg)
    assert info["state"] == "built"
    records = read_log()
    assert len(records) == 1
    rec = records[-1]
    assert rec["kind"] == "derived"
    assert rec["path"] == "data/clean/tunis.parquet"
    inputs = rec["params"]["inputs"]
    assert inputs["data/raw/power/tunis/site.json"] == file_sha256(power.cell_path("tunis", "site"))
    assert "data/raw/s2/tunis.csv" in inputs
    assert verify_log(common.PATHS["provenance_log"], root=repo, keys_dir=common.PATHS["keys"]).ok

    _, again = align.build_location("tunis", cfg)       # same inputs: nothing to do
    assert again["state"] == "unchanged"
    assert len(read_log()) == 1

    write_s2(rows=S2_ROWS[:2])                            # input changed: rebuild
    _, changed = align.build_location("tunis", cfg)
    assert changed["state"] == "built"
    assert len(read_log()) == 2

    _, forced = align.build_location("tunis", cfg, force=True)
    assert forced["state"] == "built"
    assert len(read_log()) == 3


def test_missing_site_raises(repo, cfg):
    with pytest.raises(FileNotFoundError):
        align.build_clean("kairouan", cfg)


def test_unknown_location_raises(repo, cfg):
    with pytest.raises(KeyError):
        align.build_clean("atlantis", cfg)


def test_list_locations(cfg):
    assert align.list_locations(cfg=cfg) == ["tunis", "kairouan"]
    assert align.list_locations("train", cfg=cfg) == ["tunis"]
    assert align.list_locations("test", cfg=cfg) == ["kairouan"]
    with pytest.raises(ValueError):
        align.list_locations("validation", cfg=cfg)


def test_synthetic_frame_passes_validator():
    align.validate_c1(make_synthetic_frame())