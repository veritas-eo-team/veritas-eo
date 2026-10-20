"""Tests for the signed hash-chained provenance log (module 1)."""
from __future__ import annotations

import base64
import fnmatch
import json
from datetime import date

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.common import PATHS, ROOT
from src.trust import verify as verify_mod
from src.trust.provenance import (
    GENESIS_HASH, PASSPHRASE_ENV, PRIVATE_KEY_NAME, PUBLIC_KEY_NAME, RECORD_FIELDS,
    append_record, compute_record_hash, file_sha256, init_keys, load_private_key,
    load_public_keys, public_key_id, read_log,
)
from src.trust.verify import bench_overhead, tamper_demo, verify_log


def _write_log(records):
    lines = [json.dumps(r, sort_keys=True, separators=(",", ":")) for r in records]
    PATHS["provenance_log"].write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


@pytest.fixture
def chain(tmp_paths):
    """Four recorded files under data/raw/power/demo/ in a temporary project."""
    init_keys()
    folder = PATHS["raw_power"] / "demo"
    folder.mkdir(parents=True, exist_ok=True)
    files, records = [], []
    for i in range(4):
        path = folder / f"cell_{i}.json"
        path.write_text(json.dumps({"cell": i, "value": 10.5 + i}), encoding="utf-8")
        files.append(path)
        kind = "download" if i < 3 else "derived"
        records.append(append_record(path, kind, source_url=f"https://example.invalid/{i}",
                                     params={"cell": i}))
    return files, records


# ------------------------------------------------------------- clean chain

def test_clean_chain_verifies(chain):
    files, records = chain
    result = verify_log()
    assert result.ok
    assert result.n_records == 4
    assert result.first_bad_seq is None
    assert result.head_hash == records[-1]["record_hash"]

    stored = read_log()
    assert stored == records
    assert stored[0]["prev_hash"] == GENESIS_HASH
    for i, rec in enumerate(stored):
        assert set(rec) == set(RECORD_FIELDS)
        assert rec["seq"] == i + 1
        assert rec["path"] == f"data/raw/power/demo/cell_{i}.json"
        assert rec["sha256"] == file_sha256(files[i])
        assert rec["created_at"].endswith("Z")
        assert len(rec["public_key_id"]) == 16
        assert rec["record_hash"] == compute_record_hash(rec)
        if i:
            assert rec["prev_hash"] == stored[i - 1]["record_hash"]


def test_signature_is_over_record_hash_bytes(chain):
    rec = chain[1][2]
    key = load_public_keys()[rec["public_key_id"]]
    key.verify(base64.b64decode(rec["signature"]), bytes.fromhex(rec["record_hash"]))


def test_empty_log_verifies(tmp_paths):
    result = verify_log()
    assert result.ok
    assert result.n_records == 0


# ---------------------------------------------------------------- tampering

def test_edited_file_is_caught(chain):
    files, _ = chain
    files[1].write_text(json.dumps({"cell": 1, "value": 99.9}), encoding="utf-8")
    result = verify_log()
    assert not result.ok
    assert result.first_bad_seq == 2
    assert "sha256" in result.reason


def test_edited_record_field_is_caught(chain):
    records = read_log()
    records[1]["params"] = {"cell": 99}
    _write_log(records)
    result = verify_log()
    assert result.first_bad_seq == 2
    assert "record_hash" in result.reason


def test_rehashed_record_still_fails_signature(chain):
    records = read_log()
    records[1]["params"] = {"cell": 99}
    records[1]["record_hash"] = compute_record_hash(records[1])
    _write_log(records)
    result = verify_log()
    assert result.first_bad_seq == 2
    assert "signature" in result.reason


def test_deleted_record_is_caught(chain):
    records = read_log()
    del records[2]
    _write_log(records)
    result = verify_log()
    assert result.first_bad_seq == 3
    assert "seq" in result.reason


def test_deleted_and_renumbered_is_caught(chain):
    records = read_log()
    del records[2]
    for i, rec in enumerate(records):
        rec["seq"] = i + 1
    _write_log(records)
    result = verify_log()
    assert result.first_bad_seq == 3
    assert "prev_hash" in result.reason


def test_reordered_records_are_caught(chain):
    records = read_log()
    records[1], records[2] = records[2], records[1]
    _write_log(records)
    result = verify_log()
    assert result.first_bad_seq == 2


def test_wrong_public_key_is_caught(chain):
    other = Ed25519PrivateKey.generate().public_key()
    (PATHS["keys"] / PUBLIC_KEY_NAME).write_bytes(other.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    result = verify_log()
    assert result.first_bad_seq == 1
    assert "key" in result.reason


def test_untrusted_key_is_caught(chain):
    result = verify_log(trusted_key_ids=["0" * 16])
    assert result.first_bad_seq == 1
    assert verify_log(trusted_key_ids=[public_key_id()]).ok


def test_malformed_line_is_caught(chain):
    lines = PATHS["provenance_log"].read_text(encoding="utf-8").splitlines()
    lines[2] = "{not json"
    PATHS["provenance_log"].write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    result = verify_log()
    assert result.first_bad_seq == 3
    assert "JSON" in result.reason


def test_truncation_needs_expected_head(chain):
    records = read_log()
    head = records[-1]["record_hash"]
    _write_log(records[:-1])
    assert verify_log().ok
    result = verify_log(expected_head=head)
    assert not result.ok
    assert "truncated" in result.reason


# ------------------------------------------------------- normal operations

def test_redownload_supersedes_old_hash(chain):
    files, _ = chain
    files[0].write_text("new content from a fresh download", encoding="utf-8")
    assert verify_log().first_bad_seq == 1
    append_record(files[0], "download", source_url="https://example.invalid/0")
    assert verify_log().ok


def test_missing_file_handling(chain):
    files, _ = chain
    files[2].unlink()
    result = verify_log()
    assert result.first_bad_seq == 3
    assert "missing" in result.reason
    assert verify_log(allow_missing=True).ok
    assert verify_log(check_files=False).ok


def test_key_rotation_keeps_old_records_valid(chain):
    files, _ = chain
    old_id = public_key_id()
    new_id = init_keys(force=True)
    assert new_id != old_id
    assert (PATHS["keys"] / f"ed25519_public_{old_id}.pem").exists()
    rec = append_record(files[0], "derived", params={"step": "after rotation"})
    assert rec["public_key_id"] == new_id
    assert verify_log().ok


def test_append_creates_keys_on_first_use(tmp_paths):
    path = PATHS["clean"] / "a.parquet"
    path.write_bytes(b"abc")
    rec = append_record(path, "derived")
    assert (PATHS["keys"] / PRIVATE_KEY_NAME).exists()
    assert (PATHS["keys"] / PUBLIC_KEY_NAME).exists()
    assert rec["seq"] == 1
    assert rec["prev_hash"] == GENESIS_HASH


def test_reading_record_without_file(tmp_paths):
    rec = append_record(None, "reading", sha256="ab" * 32, params={"device": "s1"})
    assert rec["path"] is None
    assert verify_log().ok


def test_params_are_normalised(tmp_paths):
    path = PATHS["clean"] / "b.parquet"
    path.write_bytes(b"xyz")
    rec = append_record(path, "derived", params={"day": date(2024, 1, 1), "cells": ("a", "b")})
    assert rec["params"] == {"day": "2024-01-01", "cells": ["a", "b"]}
    assert verify_log().ok
    with pytest.raises(ValueError):
        append_record(path, "derived", params={"x": float("nan")})


def test_append_validation(tmp_paths, tmp_path_factory):
    path = PATHS["clean"] / "c.parquet"
    path.write_bytes(b"x")
    with pytest.raises(ValueError):
        append_record(path, "bogus")
    with pytest.raises(ValueError):
        append_record(None, "reading")
    with pytest.raises(ValueError):
        append_record(None, "reading", sha256="xyz")
    with pytest.raises(ValueError):
        append_record(path, "derived", sha256="0" * 64)
    outside = tmp_path_factory.mktemp("outside") / "x.txt"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        append_record(outside, "download")


def test_passphrase_protected_key(tmp_paths, monkeypatch):
    monkeypatch.setenv(PASSPHRASE_ENV, "correct horse battery")
    init_keys()
    assert b"ENCRYPTED" in (PATHS["keys"] / PRIVATE_KEY_NAME).read_bytes()
    path = PATHS["clean"] / "d.parquet"
    path.write_bytes(b"data")
    append_record(path, "derived")
    assert verify_log().ok


# ------------------------------------------------------------ secrets

def test_private_key_never_written_out(chain, caplog):
    raw = load_private_key().private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption(),
    )
    secrets = (raw.hex(), base64.b64encode(raw).decode("ascii"))
    text = PATHS["provenance_log"].read_text(encoding="utf-8")
    assert "PRIVATE" not in text
    for secret in secrets:
        assert secret not in text
    with caplog.at_level("DEBUG"):
        append_record(chain[0][0], "derived")
        verify_log()
        tamper_demo()
    assert "PRIVATE KEY" not in caplog.text
    for secret in secrets:
        assert secret not in caplog.text


def test_gitignore_hides_private_key():
    patterns = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "keys/*private*" in patterns
    assert fnmatch.fnmatch(f"keys/{PRIVATE_KEY_NAME}", "keys/*private*")
    assert not fnmatch.fnmatch(f"keys/{PUBLIC_KEY_NAME}", "keys/*private*")


# --------------------------------------------------- demo, bench and CLI

def test_tamper_demo_without_real_data(tmp_paths):
    result = tamper_demo()
    assert not result.ok
    assert result.first_bad_seq == 1
    assert not PATHS["provenance_log"].exists()
    assert not (PATHS["keys"] / PRIVATE_KEY_NAME).exists()


def test_tamper_demo_leaves_real_data_untouched(chain):
    files, _ = chain
    watched = files + [PATHS["provenance_log"]]
    before = {path: path.read_bytes() for path in watched}
    result = tamper_demo()
    assert not result.ok
    assert result.first_bad_seq == 4
    assert {path: path.read_bytes() for path in watched} == before
    assert verify_log().ok


def test_bench_overhead_small():
    stats = bench_overhead(n=20, hash_mb=1)
    for key in ("sign_ms", "verify_ms", "hash_ms_per_mb", "append_ms",
                "chain_verify_ms_per_record", "bytes_per_record"):
        assert stats[key] > 0
    assert stats["n"] == 20


def test_cli_exit_codes_and_no_absolute_paths(chain, capsys):
    files, _ = chain
    assert verify_mod.main([]) == 0
    files[0].write_text("tampered", encoding="utf-8")
    assert verify_mod.main([]) == 1
    assert verify_mod.main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "FAILED at record 1" in out
    assert str(PATHS["root"]) not in out
    assert str(PATHS["root"].resolve()) not in out
