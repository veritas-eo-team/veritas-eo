"""Tests for src.common and src.synthetic (module 0)."""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import pytest
import yaml

from src.common import (
    EPS, PATHS, ROOT, build_paths, code_version, deep_merge, from_root,
    get_location, get_logger, load_config, rel_to_root, set_seed,
)
from src.synthetic import (
    CLEAR_CLOUD_MAX, CONTEXT_VARS, REVISIT_DAYS, TARGET_VAR, make_synthetic_frame,
)

C1_ATTRS = ("location", "lat", "lon", "climate", "split", "target_var", "units")


def expected_columns(n_neighbors: int) -> list[str]:
    return (
        ["target"]
        + [f"nb_{i}" for i in range(n_neighbors)]
        + [f"ctx_{var}" for var in CONTEXT_VARS]
        + ["ndvi", "cloud", "days_since_clear"]
    )


# ---------------------------------------------------------------- config

def test_locations_count_and_split(cfg):
    locations = cfg["locations"]
    assert len(locations) == 15
    splits = [loc["split"] for loc in locations]
    assert splits.count("train") == 10
    assert splits.count("test") == 5
    assert len({loc["id"] for loc in locations}) == 15


def test_core_settings(cfg):
    assert cfg["target"] == TARGET_VAR
    assert list(cfg["context_vars"]) == list(CONTEXT_VARS)
    assert cfg["start"] == "2018-01-01"
    assert cfg["end"] == "2024-12-31"
    assert cfg["test_year"] == 2024
    assert cfg["eval"]["attack_window"] == ["02-01", "09-30"]
    policy = cfg["policy"]
    assert policy["accept"] > policy["deweight"] > policy["verify"]
    for section in ("context_model", "step", "drift", "replay", "consistency", "fusion"):
        assert isinstance(cfg[section], dict)


def test_eps_matches_config(cfg):
    assert cfg["eps"] == pytest.approx(EPS)


def test_get_location(cfg):
    first = cfg["locations"][0]
    assert get_location(cfg, first["id"]) == first
    with pytest.raises(KeyError):
        get_location(cfg, "nowhere")


def test_tuned_overrides_settings(tmp_paths):
    PATHS["tuned"].write_text(yaml.safe_dump({"step": {"h": 7.5}}), encoding="utf-8")
    cfg = load_config()
    base = yaml.safe_load(PATHS["settings"].read_text(encoding="utf-8"))
    assert cfg["step"]["h"] == 7.5
    assert cfg["step"]["k"] == base["step"]["k"]
    assert len(cfg["locations"]) == 15


def test_empty_tuned_is_ignored(tmp_paths):
    PATHS["tuned"].write_text("", encoding="utf-8")
    assert len(load_config()["locations"]) == 15


def test_tuned_cannot_change_locations(tmp_paths):
    PATHS["tuned"].write_text(yaml.safe_dump({"locations": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config()


def test_invalid_config_rejected(tmp_paths):
    bad = yaml.safe_load(PATHS["settings"].read_text(encoding="utf-8"))
    bad["locations"][1]["id"] = bad["locations"][0]["id"]
    PATHS["settings"].write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config()


def test_deep_merge_does_not_mutate():
    base = {"x": {"y": 1, "z": [1, 2]}, "k": 1}
    merged = deep_merge(base, {"x": {"y": 2}, "n": 3})
    assert merged == {"x": {"y": 2, "z": [1, 2]}, "k": 1, "n": 3}
    merged["x"]["z"].append(3)
    assert base == {"x": {"y": 1, "z": [1, 2]}, "k": 1}


def test_config_has_no_identifying_strings():
    text = (ROOT / "config" / "settings.yaml").read_text(encoding="utf-8")
    assert "@" not in text
    assert not re.search(r"(/home/|/Users/|[A-Za-z]:\\Users)", text)


# ----------------------------------------------------------------- paths

def test_build_paths_covers_layout():
    paths = build_paths(ROOT)
    for key in ("config", "raw_power", "raw_s2", "clean", "provenance",
                "keys", "models", "results", "report"):
        assert key in paths
    assert paths["settings"] == ROOT / "config" / "settings.yaml"
    assert paths["provenance_log"] == ROOT / "data" / "provenance" / "log.jsonl"


def test_paths_created_on_access(tmp_paths):
    assert PATHS["models"].is_dir()
    log = PATHS["provenance_log"]
    assert log.parent.is_dir()
    assert not log.exists()


def test_rel_to_root_round_trip(tmp_paths):
    target = PATHS["clean"] / "x.parquet"
    rel = rel_to_root(target)
    assert rel == "data/clean/x.parquet"
    assert from_root(rel) == target
    with pytest.raises(ValueError):
        rel_to_root(PATHS["root"].parent / "outside.txt")


# ------------------------------------------------------- misc helpers

def test_code_version_format():
    version = code_version()
    assert version == "unknown" or re.fullmatch(r"[0-9a-f]{4,40}", version)


def test_set_seed_is_reproducible():
    rng = set_seed(3)
    a, b = np.random.rand(), rng.random()
    rng = set_seed(3)
    assert np.random.rand() == a
    assert rng.random() == b


def test_logger_adds_no_duplicate_handlers():
    get_logger("module0.a")
    count = len(logging.getLogger().handlers)
    get_logger("module0.b")
    get_logger("module0.a")
    assert len(logging.getLogger().handlers) == count


# -------------------------------------------------------------- synthetic

def test_frame_matches_c1(frame):
    assert list(frame.columns) == expected_columns(4)
    idx = frame.index
    assert isinstance(idx, pd.DatetimeIndex)
    assert idx.name == "date"
    assert idx.tz is None
    assert idx.is_unique and idx.is_monotonic_increasing
    assert (idx[1:] - idx[:-1] == pd.Timedelta(days=1)).all()
    assert idx[0] == pd.Timestamp("2018-01-01")
    assert idx[-1] == pd.Timestamp("2024-12-31")
    assert (frame.dtypes == np.float64).all()
    for key in C1_ATTRS:
        assert key in frame.attrs
    assert frame.attrs["split"] == "train"
    assert frame.attrs["target_var"] == TARGET_VAR


def test_target_positive_and_below_clearsky(frame):
    target = frame["target"]
    assert target.notna().all()
    assert (target > 0).all()
    ratio = target / frame["ctx_CLRSKY_SFC_SW_DWN"]
    assert (ratio > 1.03).mean() < 0.01
    assert ratio.quantile(0.95) > 0.8


def test_neighbours_informative_not_identical(frame):
    nb = frame[[f"nb_{i}" for i in range(4)]]
    corr = np.corrcoef(np.log(frame["target"]), np.log(nb.mean(axis=1)))[0, 1]
    assert corr > 0.6
    for col in nb:
        assert not np.allclose(nb[col], frame["target"])


def test_sentinel_columns(frame):
    acquired = frame["cloud"].notna()
    assert 0.18 < acquired.mean() < 0.22
    cloud = frame["cloud"].dropna()
    assert cloud.between(0.0, 1.0).all()
    clearness = frame["target"] / frame["ctx_CLRSKY_SFC_SW_DWN"]
    assert np.corrcoef(cloud, clearness[acquired])[0, 1] < -0.8
    conflict = (frame["cloud"] >= 0.7) & (clearness >= 0.8)
    assert not conflict.any()


def test_ndvi_forward_fill_is_causal(frame):
    ndvi, cloud, dsc = frame["ndvi"], frame["cloud"], frame["days_since_clear"]
    clear_acq = cloud.notna() & (cloud <= CLEAR_CLOUD_MAX)
    changed = ndvi.notna() & (ndvi != ndvi.shift(1))
    assert (~changed | clear_acq).all()
    first = clear_acq.idxmax()
    assert ndvi.loc[:first].iloc[:-1].isna().all()
    assert ndvi.loc[first:].notna().all()
    assert (dsc[clear_acq] == 0).all()
    assert (dsc.dropna() >= 0).all()
    assert dsc.isna().equals(ndvi.isna())


def test_deterministic_and_seeded():
    a = make_synthetic_frame(seed=1)
    pd.testing.assert_frame_equal(a, make_synthetic_frame(seed=1))
    assert not a["target"].equals(make_synthetic_frame(seed=2)["target"])
    assert not a["target"].equals(make_synthetic_frame(location="other", seed=1)["target"])


@pytest.mark.parametrize("k", [0, 2])
def test_neighbour_count(k, frame):
    small = make_synthetic_frame(seed=0, n_neighbors=k)
    assert list(small.columns) == expected_columns(k)
    pd.testing.assert_series_equal(small["target"], frame["target"])


def test_short_range_and_test_split():
    f = make_synthetic_frame(start="2024-01-01", end="2024-03-31", split="test")
    assert len(f) == 91
    assert f.attrs["split"] == "test"
    assert f["cloud"].notna().sum() in (18, 19)
    assert REVISIT_DAYS == 5


def test_missing_fraction():
    f = make_synthetic_frame(seed=0, missing_frac=0.05)
    assert 0.03 < f["target"].isna().mean() < 0.07


@pytest.mark.parametrize(
    "kwargs", [{"n_neighbors": 5}, {"split": "valid"}, {"seed": -1}, {"missing_frac": 1.0}]
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        make_synthetic_frame(**kwargs)
