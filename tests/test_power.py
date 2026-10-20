"""Tests for the NASA POWER downloader (module 2). No internet is used."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from src.common import PATHS, get_location, load_config
from src.data import power
from src.data.power import (
    BASE_URL, MAX_PARAMS_PER_REQUEST, PowerAPIError, build_request_params, cell_specs,
    download_cell, download_location, fetch_json, load_cell, parse_power_json, request_url,
)
from src.trust.provenance import read_log
from src.trust.verify import verify_log

FIXTURE = Path(__file__).parent / "fixtures" / "power_sample.json"
CELLS = ["site", "nb_N", "nb_E", "nb_S", "nb_W"]


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


def fake_payload(params, same_values=False):
    lat, lon = float(params["latitude"]), float(params["longitude"])
    dates = pd.date_range(pd.Timestamp(params["start"]), pd.Timestamp(params["end"]), freq="D")
    variables = params["parameters"].split(",")
    offset = 0.0 if same_values else lat * 0.1 + lon * 0.03
    parameter = {
        var: {d.strftime("%Y%m%d"): round(5.0 + j + offset + 0.1 * i, 2) for i, d in enumerate(dates)}
        for j, var in enumerate(variables)
    }
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat, 100.0]},
        "properties": {"parameter": parameter},
        "header": {"fill_value": -999.0, "start": params["start"], "end": params["end"]},
        "messages": [],
        "parameters": {var: {"units": "kW-hr/m^2/day", "longname": var} for var in variables},
    }


class FakeSession:
    """Stands in for requests.Session. `script` lists failures to return first."""

    def __init__(self, script=None, same_values=False):
        self.calls = []
        self.script = list(script or [])
        self.same_values = same_values

    def get(self, url, params=None, timeout=None):
        assert url == BASE_URL
        self.calls.append(dict(params))
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return FakeResponse(item, None, text=f"error {item}")
        return FakeResponse(200, fake_payload(params, self.same_values))


class DroppingSession(FakeSession):
    """Returns responses that lack the last requested parameter."""

    def get(self, url, params=None, timeout=None):
        payload = super().get(url, params, timeout).json()
        payload["properties"]["parameter"].pop(params["parameters"].split(",")[-1])
        return FakeResponse(200, payload)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Block real HTTP and record sleeps instead of waiting."""
    def blocked(*args, **kwargs):
        raise RuntimeError("network access is disabled in tests")
    monkeypatch.setattr(requests.Session, "get", blocked)
    sleeps: list[float] = []
    monkeypatch.setattr(power, "_sleep", sleeps.append)
    monkeypatch.setattr(power, "_last_request", 0.0)
    return sleeps


@pytest.fixture
def small_cfg(tmp_paths):
    cfg = load_config()
    cfg["start"], cfg["end"] = "2024-01-01", "2024-01-10"
    return cfg


# ---------------------------------------------------------------- parsing

def test_parse_fixture():
    df, units = parse_power_json(json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert list(df.columns) == ["ALLSKY_SFC_SW_DWN", "T2M"]
    assert df.index.name == "date"
    assert len(df) == 5
    assert df.index[0] == pd.Timestamp("2024-01-01")
    assert (df.index[1:] - df.index[:-1] == pd.Timedelta(days=1)).all()
    assert np.isnan(df.loc["2024-01-03", "ALLSKY_SFC_SW_DWN"])
    assert df["T2M"].notna().all()
    assert (df.dtypes == np.float64).all()
    assert units == {"ALLSKY_SFC_SW_DWN": "kW-hr/m^2/day", "T2M": "C"}
    assert df.attrs["lat"] == 36.8
    assert df.attrs["lon"] == 10.18


def test_fill_values_become_nan():
    obj = {"properties": {"parameter": {"X": {
        "20240101": -999.0, "20240102": -990.0, "20240103": -989.9,
        "20240104": None, "20240105": 3.2}}}}
    df, units = parse_power_json(obj)
    x = df["X"].tolist()
    assert np.isnan(x[0]) and np.isnan(x[1]) and np.isnan(x[3])
    assert x[2] == pytest.approx(-989.9)
    assert x[4] == pytest.approx(3.2)
    assert units == {"X": None}


def test_parse_rejects_error_payload():
    with pytest.raises(ValueError):
        parse_power_json({"messages": ["bad request"]})


# ---------------------------------------------------------- request shape

def test_cell_specs(small_cfg):
    specs = cell_specs(get_location(small_cfg, "tunis"), small_cfg)
    assert [s["cell_id"] for s in specs] == CELLS
    site = specs[0]
    assert site["parameters"] == [small_cfg["target"]] + list(small_cfg["context_vars"])
    assert (site["lat"], site["lon"]) == (36.8, 10.18)
    by_id = {s["cell_id"]: s for s in specs}
    assert (by_id["nb_N"]["lat"], by_id["nb_N"]["lon"]) == (37.8, 10.18)
    assert (by_id["nb_E"]["lat"], by_id["nb_E"]["lon"]) == (36.8, 11.18)
    assert (by_id["nb_S"]["lat"], by_id["nb_S"]["lon"]) == (35.8, 10.18)
    assert (by_id["nb_W"]["lat"], by_id["nb_W"]["lon"]) == (36.8, 9.18)
    for cell in CELLS[1:]:
        assert by_id[cell]["parameters"] == [small_cfg["target"]]


def test_cell_specs_wrap_and_clip(small_cfg):
    specs = {s["cell_id"]: s for s in cell_specs({"lat": 89.8, "lon": 179.5}, small_cfg)}
    assert specs["nb_E"]["lon"] == -179.5
    assert specs["nb_N"]["lat"] == 90.0


def test_build_request_params():
    params = build_request_params(36.8, 10.18, ["T2M", "RH2M"], "2024-01-01", "2024-12-31")
    assert params == {
        "parameters": "T2M,RH2M", "community": "AG", "latitude": "36.8000",
        "longitude": "10.1800", "start": "20240101", "end": "20241231", "format": "JSON",
    }
    url = request_url(params)
    assert url.startswith(BASE_URL + "?")
    assert "community=AG" in url


# ------------------------------------------------- caching and provenance

def test_download_caches_and_records_provenance(small_cfg):
    session = FakeSession()
    summary = download_location("tunis", small_cfg, session=session)
    assert len(session.calls) == 5
    assert summary["cells"] == {cell: "downloaded" for cell in CELLS}
    assert summary["identical_neighbours"] == []
    for cell in CELLS:
        assert (PATHS["raw_power"] / "tunis" / f"{cell}.json").exists()

    records = read_log()
    assert len(records) == 5
    site = records[0]
    assert site["kind"] == "download"
    assert site["path"] == "data/raw/power/tunis/site.json"
    assert site["source_url"].startswith(BASE_URL)
    assert site["params"]["cell_id"] == "site"
    assert site["params"]["parameters"] == [small_cfg["target"]] + list(small_cfg["context_vars"])
    assert site["params"]["start"] == "20240101"
    assert verify_log().ok

    again = FakeSession()
    summary = download_location("tunis", small_cfg, session=again)
    assert again.calls == []
    assert set(summary["cells"].values()) == {"cached"}
    assert len(read_log()) == 5

    forced = FakeSession()
    download_location("tunis", small_cfg, force=True, session=forced)
    assert len(forced.calls) == 5
    assert len(read_log()) == 10
    assert verify_log().ok


def test_downloaded_cell_parses(small_cfg):
    download_location("tunis", small_cfg, session=FakeSession())
    df, units = load_cell("tunis", "site")
    assert list(df.columns) == [small_cfg["target"]] + list(small_cfg["context_vars"])
    assert len(df) == 10
    assert units[small_cfg["target"]] == "kW-hr/m^2/day"


def test_batching_over_20_parameters(small_cfg):
    spec = {"cell_id": "site", "lat": 36.8, "lon": 10.18, "parameters": [f"P{i}" for i in range(25)]}
    session = FakeSession()
    assert download_cell("tunis", spec, small_cfg, session=session) == "downloaded"
    assert len(session.calls) == 2
    assert len(session.calls[0]["parameters"].split(",")) == MAX_PARAMS_PER_REQUEST
    df, _ = load_cell("tunis", "site")
    assert len(df.columns) == 25
    assert read_log()[0]["params"]["requests"] == 2


def test_incomplete_response_is_rejected(small_cfg):
    spec = cell_specs(get_location(small_cfg, "tunis"), small_cfg)[0]
    with pytest.raises(PowerAPIError):
        download_cell("tunis", spec, small_cfg, session=DroppingSession())
    assert not (PATHS["raw_power"] / "tunis" / "site.json").exists()
    assert read_log() == []


def test_failed_provenance_leaves_no_file(small_cfg, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(power, "append_record", broken)
    spec = cell_specs(get_location(small_cfg, "tunis"), small_cfg)[0]
    with pytest.raises(OSError):
        download_cell("tunis", spec, small_cfg, session=FakeSession())
    assert not (PATHS["raw_power"] / "tunis" / "site.json").exists()


def test_identical_neighbours_are_reported(small_cfg, caplog):
    with caplog.at_level("WARNING"):
        summary = download_location("tunis", small_cfg, session=FakeSession(same_values=True))
    assert summary["identical_neighbours"] == CELLS[1:]
    assert "identical" in caplog.text


# --------------------------------------------------- retries and throttling

PARAMS = build_request_params(36.8, 10.18, ["T2M"], "2024-01-01", "2024-01-03")


def test_retries_then_success(offline):
    session = FakeSession(script=[503, 503])
    data = fetch_json(PARAMS, session=session, delay_s=0.0, backoff_s=2.0)
    assert "properties" in data
    assert len(session.calls) == 3
    assert offline == [2.0, 4.0]


def test_connection_error_is_retried():
    session = FakeSession(script=[requests.ConnectionError("down")])
    assert "properties" in fetch_json(PARAMS, session=session, delay_s=0.0)
    assert len(session.calls) == 2


def test_gives_up_after_three_retries():
    session = FakeSession(script=[503, 503, 503, 503])
    with pytest.raises(PowerAPIError):
        fetch_json(PARAMS, session=session, delay_s=0.0)
    assert len(session.calls) == 4


def test_client_error_is_not_retried():
    session = FakeSession(script=[422])
    with pytest.raises(PowerAPIError):
        fetch_json(PARAMS, session=session, delay_s=0.0)
    assert len(session.calls) == 1


def test_one_second_between_requests(offline, monkeypatch):
    monkeypatch.setattr(power, "_last_request", time.monotonic())
    fetch_json(PARAMS, session=FakeSession(), delay_s=1.0)
    assert len(offline) == 1
    assert 0.5 < offline[0] <= 1.0


# --------------------------------------------------------------------- CLI

def test_cli(tmp_paths, monkeypatch, capsys):
    monkeypatch.setattr(power, "_make_session", lambda: FakeSession())
    assert power.main(["--location", "tunis"]) == 0
    assert power.main(["--location", "nowhere"]) == 1
    out = capsys.readouterr().out
    assert "tunis" in out
    assert str(PATHS["root"]) not in out
    assert len(read_log()) == 5
    assert verify_log().ok
