"""Tests for src/data/sentinel2.py. No internet: token and statistics calls are mocked."""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

import src.data.sentinel2 as s2
from src.trust.provenance import read_log
from src.trust.verify import verify_log

SECRET = "test-secret-value-123"


# ---------------------------------------------------------------- helpers

def stats_block(mean, sample=22500, nodata=0):
    return {"bands": {"B0": {"stats": {
        "min": 0.0, "max": 1.0, "mean": mean, "stDev": 0.1,
        "sampleCount": sample, "noDataCount": nodata,
    }}}}


def interval(day, ndvi_clear, clear, cloud, sample=22500, nodata=0):
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return {
        "interval": {"from": f"{day}T00:00:00Z", "to": f"{nxt}T00:00:00Z"},
        "outputs": {
            "ndvi_clear": stats_block(ndvi_clear, sample, nodata),
            "clear": stats_block(clear, sample, nodata),
            "cloud": stats_block(cloud, sample, nodata),
        },
    }


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class FakeSession:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.handler(url, kwargs, len(self.calls))

    def n_calls(self, url):
        return sum(1 for u, _ in self.calls if u == url)


def token_ok():
    return FakeResponse(200, {"access_token": "tok", "expires_in": 600})


def stats_ok(kwargs):
    year = kwargs["json"]["aggregation"]["timeRange"]["from"][:4]
    return FakeResponse(200, {"status": "OK", "data": [
        interval(f"{year}-03-01", 0.24, 0.6, 0.3),
        interval(f"{year}-03-06", 0.0, 0.0, 1.0),
    ]})


def ok_handler(url, kwargs, n):
    return token_ok() if url == s2.TOKEN_URL else stats_ok(kwargs)


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def s2cfg():
    return {
        "start": "2018-01-01",
        "end": "2019-12-31",
        "locations": [
            {"id": "tunis", "lat": 36.80, "lon": 10.18, "climate": "mediterranean", "split": "train"},
        ],
        "sentinel2": {"delay_s": 0.0, "backoff_s": 0.0},
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw" / "s2"
    raw.mkdir(parents=True)
    (tmp_path / "data" / "provenance").mkdir(parents=True)
    (tmp_path / "keys").mkdir()
    monkeypatch.setattr(s2, "s2_dir", lambda: raw)
    monkeypatch.setattr(s2, "_sleep", lambda seconds: None)
    monkeypatch.setenv(s2.ENV_CLIENT_ID, "test-client")
    monkeypatch.setenv(s2.ENV_CLIENT_SECRET, SECRET)
    monkeypatch.delenv("PROVENANCE_KEY_PASSPHRASE", raising=False)
    return {
        "raw": raw,
        "prov": {
            "log_path": tmp_path / "data" / "provenance" / "log.jsonl",
            "keys_dir": tmp_path / "keys",
            "root": tmp_path,
        },
    }


# ---------------------------------------------------------------- request building

def test_bbox_is_about_3km():
    lat, lon = 36.80, 10.18
    b = s2.bbox_around(lat, lon, 3.0)
    height_km = (b[3] - b[1]) * 110.574
    width_km = (b[2] - b[0]) * 111.320 * math.cos(math.radians(lat))
    assert height_km == pytest.approx(3.0, rel=1e-6)
    assert width_km == pytest.approx(3.0, rel=1e-6)
    assert (b[0] + b[2]) / 2 == pytest.approx(lon)
    assert (b[1] + b[3]) / 2 == pytest.approx(lat)


def test_year_ranges():
    assert s2.year_ranges("2018-01-01", "2019-12-31") == [
        ("2018-01-01T00:00:00Z", "2019-01-01T00:00:00Z"),
        ("2019-01-01T00:00:00Z", "2020-01-01T00:00:00Z"),
    ]
    assert s2.year_ranges("2018-06-15", "2018-07-01") == [
        ("2018-06-15T00:00:00Z", "2018-07-02T00:00:00Z"),
    ]


def test_request_body():
    body = s2.build_request_body([10.0, 36.0, 10.1, 36.1], "a", "b", s2.s2_config({}))
    json.dumps(body)  # serialisable
    assert body["input"]["data"][0]["type"] == "sentinel-2-l2a"
    agg = body["aggregation"]
    assert agg["aggregationInterval"]["of"] == "P1D"
    assert agg["width"] == agg["height"] == 150  # 3 km at 20 m
    for cls in s2.CLOUD_SCL_CLASSES:
        assert f"s.SCL == {cls}" in agg["evalscript"]


def test_locations_accepts_list_and_dict():
    item = {"id": "tunis", "lat": 1.0, "lon": 2.0}
    assert list(s2._locations({"locations": [item]})) == ["tunis"]
    assert list(s2._locations({"locations": {"tunis": {"lat": 1.0, "lon": 2.0}}})) == ["tunis"]
    assert list(s2._locations({"locations": {"train": [item]}})) == ["tunis"]


# ---------------------------------------------------------------- parsing

def test_parse_stats():
    obj = {"data": [
        interval("2018-03-01", 0.24, 0.6, 0.3, nodata=2250),
        interval("2018-03-02", "NaN", "NaN", "NaN", nodata=22500),  # no data -> skipped
        {"interval": {"from": "2018-03-04T00:00:00Z", "to": "2018-03-05T00:00:00Z"},
         "error": {"type": "EXECUTION_ERROR"}},                    # error -> skipped
        interval("2018-03-06", 0.0, 0.0, 1.0),                      # fully cloudy
    ]}
    df = s2.parse_stats(obj)
    assert list(df.columns) == s2.COLUMNS
    assert df.index.name == "date"
    assert list(df.index) == [pd.Timestamp("2018-03-01"), pd.Timestamp("2018-03-06")]
    assert all(dt == "float64" for dt in df.dtypes)
    first = df.loc["2018-03-01"]
    assert first["ndvi_mean"] == pytest.approx(0.4)        # 0.24 / 0.6
    assert first["cloud_fraction"] == pytest.approx(0.3)
    assert first["valid_fraction"] == pytest.approx(0.9)   # 1 - 2250/22500
    cloudy = df.loc["2018-03-06"]
    assert math.isnan(cloudy["ndvi_mean"])
    assert cloudy["cloud_fraction"] == pytest.approx(1.0)
    assert cloudy["valid_fraction"] == pytest.approx(1.0)


def test_parse_rejects_non_stats_json():
    with pytest.raises(ValueError):
        s2.parse_stats({"foo": 1})


def test_load_missing_file_returns_empty(env):
    df = s2.load_s2("nowhere")
    assert df.empty
    assert list(df.columns) == s2.COLUMNS
    assert isinstance(df.index, pd.DatetimeIndex)


# ---------------------------------------------------------------- download, cache, provenance

def test_missing_credentials_returns_empty(env, s2cfg, monkeypatch):
    monkeypatch.delenv(s2.ENV_CLIENT_ID, raising=False)
    monkeypatch.delenv(s2.ENV_CLIENT_SECRET, raising=False)
    session = FakeSession(ok_handler)
    df, state = s2.download_location("tunis", s2cfg, session=session, **env["prov"])
    assert state == "no_credentials"
    assert df.empty and list(df.columns) == s2.COLUMNS
    assert session.calls == []
    assert not (env["raw"] / "tunis.csv").exists()


def test_download_caches_and_logs(env, s2cfg):
    session = FakeSession(ok_handler)
    df, state = s2.download_location("tunis", s2cfg, session=session, **env["prov"])
    assert state == "downloaded"
    assert len(df) == 4  # 2 acquisitions x 2 years
    assert session.n_calls(s2.TOKEN_URL) == 1
    assert session.n_calls(s2.STATS_URL) == 2
    path = env["raw"] / "tunis.csv"
    assert path.exists()

    records = read_log(env["prov"]["log_path"])
    assert len(records) == 1
    rec = records[-1]
    assert rec["kind"] == "download"
    assert rec["source_url"] == s2.STATS_URL
    assert str(rec["path"]).replace("\\", "/").endswith("data/raw/s2/tunis.csv")
    assert rec["params"]["location"] == "tunis"
    assert rec["params"]["requests"] == 2
    assert verify_log(
        env["prov"]["log_path"], root=env["prov"]["root"], keys_dir=env["prov"]["keys_dir"]
    ).ok

    # second call uses the cache: no new requests, same data
    df2, state2 = s2.download_location("tunis", s2cfg, session=session, **env["prov"])
    assert state2 == "cached"
    assert len(session.calls) == 3
    pd.testing.assert_frame_equal(df2, df, check_freq=False, rtol=1e-6)


def test_force_redownloads(env, s2cfg):
    session = FakeSession(ok_handler)
    s2.download_location("tunis", s2cfg, session=session, **env["prov"])
    _, state = s2.download_location("tunis", s2cfg, force=True, session=session, **env["prov"])
    assert state == "downloaded"
    assert session.n_calls(s2.STATS_URL) == 4
    assert len(read_log(env["prov"]["log_path"])) == 2


def test_retry_on_429(env, s2cfg, monkeypatch):
    waits = []
    monkeypatch.setattr(s2, "_sleep", waits.append)
    s2cfg["sentinel2"]["backoff_s"] = 2.0
    state_box = {"stats_calls": 0}

    def handler(url, kwargs, n):
        if url == s2.TOKEN_URL:
            return token_ok()
        state_box["stats_calls"] += 1
        if state_box["stats_calls"] == 1:
            return FakeResponse(429, text="slow down")
        return stats_ok(kwargs)

    _, state = s2.download_location("tunis", s2cfg, session=FakeSession(handler), **env["prov"])
    assert state == "downloaded"
    assert 2.0 in waits


def test_expired_token_is_refreshed(env, s2cfg):
    state_box = {"stats_calls": 0}

    def handler(url, kwargs, n):
        if url == s2.TOKEN_URL:
            return token_ok()
        state_box["stats_calls"] += 1
        if state_box["stats_calls"] == 1:
            return FakeResponse(401, text="expired")
        return stats_ok(kwargs)

    session = FakeSession(handler)
    _, state = s2.download_location("tunis", s2cfg, session=session, **env["prov"])
    assert state == "downloaded"
    assert session.n_calls(s2.TOKEN_URL) == 2


def test_api_failure_returns_empty(env, s2cfg):
    def handler(url, kwargs, n):
        return token_ok() if url == s2.TOKEN_URL else FakeResponse(500, text="server error")

    df, state = s2.download_location("tunis", s2cfg, session=FakeSession(handler), **env["prov"])
    assert state == "failed"
    assert df.empty and list(df.columns) == s2.COLUMNS
    assert not (env["raw"] / "tunis.csv").exists()
    assert not env["prov"]["log_path"].exists()


def test_bad_credentials_fail_without_leaking_secret(env, s2cfg, caplog):
    def handler(url, kwargs, n):
        return FakeResponse(401, text='{"error":"unauthorized_client"}')

    with caplog.at_level("DEBUG"):
        df, state = s2.download_location("tunis", s2cfg, session=FakeSession(handler), **env["prov"])
    assert state == "failed"
    assert df.empty
    assert SECRET not in caplog.text


def test_provenance_failure_removes_file(env, s2cfg, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("log unavailable")

    monkeypatch.setattr(s2, "append_record", boom)
    df, state = s2.download_location("tunis", s2cfg, session=FakeSession(ok_handler), **env["prov"])
    assert state == "failed"
    assert df.empty
    assert not (env["raw"] / "tunis.csv").exists()


def test_unknown_location_raises(env, s2cfg):
    with pytest.raises(KeyError):
        s2.download_location("atlantis", s2cfg, session=FakeSession(ok_handler), **env["prov"])