"""Verify the provenance log, run a tamper demo, and measure security overhead.

CLI: python -m src.trust.verify [--demo] [--bench] [--n N] [--no-files]
     [--allow-missing] [--trust KEY_ID] [--expected-head HASH]
Exit code 1 on failure.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.common import PATHS, get_logger
from src.trust.provenance import (
    GENESIS_HASH,
    KINDS,
    RECORD_FIELDS,
    LogFormatError,
    append_record,
    compute_record_hash,
    file_sha256,
    init_keys,
    load_public_keys,
    read_log,
)

log = get_logger(__name__)


@dataclass
class VerifyResult:
    """Outcome of verify_log. first_bad_seq is the 1-based position of the first bad record."""

    ok: bool
    n_records: int
    first_bad_seq: int | None
    reason: str
    head_hash: str | None = None


def _safe_rel(path: str) -> bool:
    """True for a relative POSIX path that stays inside the project root."""
    pure = PurePosixPath(path)
    return bool(path.strip()) and not pure.is_absolute() and ".." not in pure.parts and ":" not in path


def verify_log(
    log_path: Path | str | None = None,
    check_files: bool = True,
    *,
    root: Path | str | None = None,
    keys_dir: Path | str | None = None,
    allow_missing: bool = False,
    trusted_key_ids: Iterable[str] | None = None,
    expected_head: str | None = None,
) -> VerifyResult:
    """Check every record in order: seq, prev_hash, record_hash, signature, then the file.

    Only the most recent record for each path is compared with the file on disk,
    so a legitimate re-download (a newer record for the same path) is fine.
    trusted_key_ids pins which public keys are accepted. expected_head (a
    record_hash published earlier) detects truncation of the end of the log.
    """
    log_file = Path(log_path) if log_path is not None else PATHS["provenance_log"]
    root_dir = Path(root) if root is not None else PATHS["root"]

    try:
        records = read_log(log_file)
    except LogFormatError as exc:
        return VerifyResult(False, exc.position - 1, exc.position, exc.reason)

    keyring = load_public_keys(keys_dir)
    if trusted_key_ids is not None:
        trusted = set(trusted_key_ids)
        keyring = {key_id: key for key_id, key in keyring.items() if key_id in trusted}

    latest: dict[str, int] = {}
    for position, rec in enumerate(records, start=1):
        if isinstance(rec.get("path"), str):
            latest[rec["path"]] = position

    n = len(records)
    prev = GENESIS_HASH
    for position, rec in enumerate(records, start=1):
        def fail(reason: str) -> VerifyResult:
            return VerifyResult(False, n, position, reason)

        missing = [field for field in RECORD_FIELDS if field not in rec]
        extra = sorted(set(rec) - set(RECORD_FIELDS))
        if missing or extra:
            return fail(f"malformed record (missing {missing}, unexpected {extra})")
        if rec["kind"] not in KINDS:
            return fail(f"malformed record (unknown kind {rec['kind']!r})")
        if rec["seq"] != position:
            return fail(f"seq is {rec['seq']}, expected {position} (record deleted, inserted or reordered)")
        if rec["prev_hash"] != prev:
            return fail("prev_hash does not match the previous record's record_hash")
        if compute_record_hash(rec) != rec["record_hash"]:
            return fail("record_hash mismatch (record edited)")
        key = keyring.get(rec["public_key_id"])
        if key is None:
            return fail("unknown or untrusted public key")
        try:
            key.verify(base64.b64decode(rec["signature"], validate=True), bytes.fromhex(rec["record_hash"]))
        except (InvalidSignature, ValueError, binascii.Error, TypeError):
            return fail("bad signature")

        rel = rec["path"]
        if check_files and rel is not None and latest.get(rel) == position:
            if not isinstance(rel, str) or not _safe_rel(rel):
                return fail("unsafe path in record")
            file_path = root_dir / rel
            if not file_path.is_file():
                if not allow_missing:
                    return fail(f"file missing: {rel}")
            elif file_sha256(file_path) != rec["sha256"]:
                return fail(f"file changed since it was recorded (sha256 mismatch): {rel}")
        prev = rec["record_hash"]

    head = records[-1]["record_hash"] if records else None
    if expected_head is not None and expected_head not in {r["record_hash"] for r in records}:
        return VerifyResult(False, n, n + 1, "expected head not found (log truncated or rewritten)", head)
    return VerifyResult(True, n, None, "ok" if records else "empty log", head)


def _tamper_file(path: Path) -> int:
    """Change one digit in the file (or append one byte). Returns the byte offset."""
    data = bytearray(path.read_bytes())
    for offset, byte in enumerate(data):
        if 48 <= byte <= 57:
            data[offset] = 48 + (byte - 48 + 1) % 10
            path.write_bytes(bytes(data))
            return offset
    data += b"0"
    path.write_bytes(bytes(data))
    return len(data) - 1


def tamper_demo() -> VerifyResult:
    """Copy a recorded file and the log to a temp folder, edit one value in the copy, verify.

    Uses the most recent recorded file that exists. If there is none, builds a
    small self-contained example with a temporary key. Real data is only read.
    Returns the result after tampering (ok=False means the edit was caught).
    """
    real_log = PATHS["provenance_log"]
    try:
        records = read_log(real_log)
    except LogFormatError:
        records = []
    candidate = None
    for rec in reversed(records):
        rel = rec.get("path")
        if isinstance(rel, str) and _safe_rel(rel) and (PATHS["root"] / rel).is_file():
            candidate = rec
            break

    with tempfile.TemporaryDirectory(prefix="tamper_demo_") as tmp_name:
        tmp = Path(tmp_name)
        log_copy = tmp / "log.jsonl"
        if candidate is not None:
            rel = candidate["path"]
            shutil.copy2(real_log, log_copy)
            target = tmp / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PATHS["root"] / rel, target)
            keys_dir: Path = PATHS["keys"]
        else:
            rel = "data/demo/sample.csv"
            keys_dir = tmp / "keys"
            init_keys(keys_dir=keys_dir)
            target = tmp / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "date,value\n2024-06-01,6.42\n2024-06-02,6.38\n2024-06-03,5.97\n",
                encoding="utf-8", newline="\n",
            )
            append_record(target, "download", source_url="https://example.invalid/demo",
                          params={"demo": True}, log_path=log_copy, keys_dir=keys_dir, root=tmp)

        options = {"root": tmp, "keys_dir": keys_dir, "allow_missing": True}
        before = verify_log(log_copy, **options)
        offset = _tamper_file(target)
        after = verify_log(log_copy, **options)

    log.info("tamper demo on a temporary copy of %s", rel)
    log.info("before edit: %s", before.reason)
    log.info("edited one byte at offset %d; after edit: record %s, %s",
             offset, after.first_bad_seq, after.reason)
    return after


def bench_overhead(n: int = 1000, hash_mb: int = 8) -> dict[str, float]:
    """Time signing, verification, hashing and appending in a temp folder (real log untouched)."""
    if n < 1 or hash_mb < 1:
        raise ValueError("n and hash_mb must be at least 1")
    key = Ed25519PrivateKey.generate()
    public = key.public_key()
    digests = [hashlib.sha256(i.to_bytes(8, "big")).digest() for i in range(n)]

    start = time.perf_counter()
    signatures = [key.sign(digest) for digest in digests]
    sign_ms = (time.perf_counter() - start) * 1000 / n

    start = time.perf_counter()
    for digest, signature in zip(digests, signatures):
        public.verify(signature, digest)
    verify_ms = (time.perf_counter() - start) * 1000 / n

    with tempfile.TemporaryDirectory(prefix="prov_bench_") as tmp_name:
        tmp = Path(tmp_name)
        blob = tmp / "blob.bin"
        blob.write_bytes(bytes(hash_mb * (1 << 20)))
        start = time.perf_counter()
        file_sha256(blob)
        hash_ms_per_mb = (time.perf_counter() - start) * 1000 / hash_mb

        keys_dir = tmp / "keys"
        init_keys(keys_dir=keys_dir)
        log_path = tmp / "log.jsonl"
        start = time.perf_counter()
        for digest in digests:
            append_record(None, "reading", sha256=digest.hex(), params={"bench": True},
                          log_path=log_path, keys_dir=keys_dir)
        append_ms = (time.perf_counter() - start) * 1000 / n
        bytes_per_record = log_path.stat().st_size / n

        start = time.perf_counter()
        result = verify_log(log_path, check_files=False, keys_dir=keys_dir)
        chain_verify_ms = (time.perf_counter() - start) * 1000 / n
        if not result.ok:
            raise RuntimeError(f"benchmark log failed verification: {result.reason}")

    return {
        "n": n,
        "sign_ms": round(sign_ms, 4),
        "verify_ms": round(verify_ms, 4),
        "hash_ms_per_mb": round(hash_ms_per_mb, 4),
        "append_ms": round(append_ms, 4),
        "chain_verify_ms_per_record": round(chain_verify_ms, 4),
        "bytes_per_record": round(bytes_per_record, 1),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Verify the signed provenance log.")
    parser.add_argument("--demo", action="store_true", help="run the tamper demo on temporary copies")
    parser.add_argument("--bench", action="store_true", help="measure signing, verification and hashing cost")
    parser.add_argument("--n", type=int, default=1000, help="records for --bench (default 1000)")
    parser.add_argument("--no-files", action="store_true", help="check the chain only, not the files")
    parser.add_argument("--allow-missing", action="store_true", help="do not fail on absent recorded files")
    parser.add_argument("--trust", action="append", metavar="KEY_ID", help="accept only these key ids (repeatable)")
    parser.add_argument("--expected-head", metavar="HASH", help="fail if this record_hash is not in the log")
    args = parser.parse_args(argv)

    code = 0
    if args.bench:
        print(json.dumps(bench_overhead(args.n), indent=2))
    if args.demo:
        result = tamper_demo()
        if result.ok:
            print("DEMO FAILED: the edited copy still verified")
            code = 1
        else:
            print(f"Tamper demo: edit caught at record {result.first_bad_seq} ({result.reason})")
    if not (args.bench or args.demo):
        result = verify_log(
            check_files=not args.no_files,
            allow_missing=args.allow_missing,
            trusted_key_ids=args.trust,
            expected_head=args.expected_head,
        )
        if result.ok:
            print(f"OK: {result.n_records} records verified; head {result.head_hash}")
        else:
            print(f"FAILED at record {result.first_bad_seq}: {result.reason}")
            code = 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
