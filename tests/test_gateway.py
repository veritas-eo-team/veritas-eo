"""Tests for src/trust/gateway.py (offline, temporary keys and logs)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import src.trust.gateway as gw
from src.trust.provenance import read_log
from src.trust.verify import verify_log


@pytest.fixture
def setup(tmp_path):
    key = Ed25519PrivateKey.generate()
    sensor = gw.Sensor("station-1", key)
    gateway = gw.Gateway({"station-1": key.public_key()}, log_path=tmp_path / "readings.jsonl",
                         keys_dir=tmp_path / "keys")
    return key, sensor, gateway, tmp_path


def test_normal_readings_are_accepted_and_seq_increases(setup):
    _, sensor, gateway, _ = setup
    m1, m2 = sensor.sign_reading("2024-05-01", 6.2), sensor.sign_reading("2024-05-02", 6.3)
    assert (m1["seq"], m2["seq"]) == (1, 2)
    assert set(m1) == set(gw.MESSAGE_FIELDS)
    assert gateway.accept(m1) == (True, "accepted")
    assert gateway.accept(m2) == (True, "accepted")


def test_altered_value_is_rejected(setup):
    _, sensor, gateway, _ = setup
    msg = sensor.sign_reading("2024-05-01", 6.2)
    msg["value"] = round(msg["value"] * 1.12, 6)
    assert gateway.accept(msg) == (False, "bad signature")


def test_replayed_message_is_rejected(setup):
    _, sensor, gateway, _ = setup
    msg = sensor.sign_reading("2024-05-01", 6.2)
    assert gateway.accept(msg)[0]
    assert gateway.accept(dict(msg)) == (False, "sequence not increasing")


def test_unknown_key_and_unknown_device_are_rejected(setup):
    _, _, gateway, _ = setup
    forged = gw.Sensor("station-1", Ed25519PrivateKey.generate(), start_seq=50).sign_reading("2024-05-01", 6.2)
    assert gateway.accept(forged) == (False, "bad signature")
    stranger = gw.Sensor("station-9", Ed25519PrivateKey.generate()).sign_reading("2024-05-01", 6.2)
    assert gateway.accept(stranger) == (False, "unknown device")


def test_old_timestamp_and_reused_nonce_are_rejected(setup):
    key, sensor, gateway, _ = setup
    late = gw.Sensor("station-1", key, clock=lambda: time.time() - 3600.0, start_seq=10)
    assert gateway.accept(late.sign_reading("2024-05-01", 6.2)) == (False, "timestamp outside window")
    fixed = gw.Sensor("station-1", key, nonce_fn=lambda: "same-nonce", start_seq=20)
    assert gateway.accept(fixed.sign_reading("2024-05-01", 6.2))[0]
    assert gateway.accept(fixed.sign_reading("2024-05-02", 6.3)) == (False, "nonce reused")


def test_malformed_message_is_rejected(setup):
    _, sensor, gateway, _ = setup
    msg = sensor.sign_reading("2024-05-01", 6.2)
    del msg["nonce"]
    assert gateway.accept(msg) == (False, "malformed message")
    assert gateway.accept("not a message") == (False, "malformed message")


def test_compromised_key_with_bias_is_accepted(setup):
    _, sensor, gateway, _ = setup
    assert gateway.accept(sensor.sign_reading("2024-05-01", 6.0 * 1.12)) == (True, "accepted")


def test_readings_log_is_a_valid_signed_chain(setup):
    _, sensor, gateway, tmp_path = setup
    for day in range(5):
        gateway.accept(sensor.sign_reading(pd.Timestamp("2024-05-01") + pd.Timedelta(days=day), 6.0 + day))
    records = read_log(tmp_path / "readings.jsonl")
    assert len(records) == 5 and {r["kind"] for r in records} == {"reading"}
    assert all(r["path"] is None for r in records)
    assert records[0]["params"]["device_id"] == "station-1"
    assert verify_log(tmp_path / "readings.jsonl", check_files=False, keys_dir=tmp_path / "keys").ok


def test_stream_series_with_tampering_in_transit(setup):
    _, sensor, gateway, _ = setup
    index = pd.date_range("2024-05-01", periods=30, freq="D", name="date")
    series = pd.Series(np.linspace(5.0, 7.0, 30), index=index)
    series.iloc[10] = np.nan
    bad_days = {index[3], index[7], index[20]}

    def tamper(date, msg):
        if date in bad_days:
            msg["value"] = round(msg["value"] * 1.12, 6)
        return msg

    accepted, rejections = gw.stream_series(series, sensor, gateway, tamper)
    assert len(rejections) == 3 and set(rejections["reason"]) == {"bad signature"}
    assert accepted["accepted"].sum() == 26
    assert accepted.loc[list(bad_days), "value"].isna().all()
    np.testing.assert_allclose(accepted.loc[index[0], "value"], 5.0)


def test_demo_attacks_behave_as_stated():
    result = gw.demo_attacks()
    assert result["as_expected"].all()
    assert result.attrs["log_ok"]
    by_case = result.set_index("case")
    assert by_case.loc["compromised sensor, valid key, +12% bias", "accepted"]
    for name in ("value altered in transit (+12%)", "old message replayed",
                 "forged with an unknown key", "unregistered device", "delayed message (1 h old)"):
        assert not by_case.loc[name, "accepted"]


def test_bench_gateway():
    b = gw.bench_gateway(n=20)
    assert b["sign_ms"] > 0 and b["check_ms"] > 0 and b["check_and_log_ms"] > 0
    assert b["bytes_per_logged_reading"] > 100