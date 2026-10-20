"""Signed, hash-chained provenance log (contract C6).

Each line of data/provenance/log.jsonl is one JSON record. record_hash is the
SHA-256 of the canonical JSON of every field except record_hash and signature.
Each record stores the previous record_hash, and signature is an Ed25519
signature over the 32 raw bytes of record_hash (bytes.fromhex(record_hash)).
Editing, deleting or reordering any record therefore breaks the chain.

Never put secrets in source_url or params: they are stored in clear text.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from src.common import PATHS, code_version, get_logger, rel_to_root

log = get_logger(__name__)

KINDS: tuple[str, ...] = ("download", "derived", "reading")
GENESIS_HASH = "0" * 64
PRIVATE_KEY_NAME = "ed25519_private.pem"
PUBLIC_KEY_NAME = "ed25519_public.pem"
PASSPHRASE_ENV = "PROVENANCE_KEY_PASSPHRASE"
RECORD_FIELDS: tuple[str, ...] = (
    "seq", "kind", "path", "source_url", "params", "created_at", "sha256",
    "code_version", "prev_hash", "record_hash", "signature", "public_key_id",
)
_NOT_HASHED: tuple[str, ...] = ("record_hash", "signature")
_CHUNK = 1 << 20
_HEX = frozenset("0123456789abcdef")
_private_cache: dict[str, Ed25519PrivateKey] = {}


class LogFormatError(ValueError):
    """A log line that is not a JSON object. `position` is its 1-based record number."""

    def __init__(self, position: int, reason: str) -> None:
        super().__init__(f"record {position}: {reason}")
        self.position = position
        self.reason = reason


# ------------------------------------------------------------------ helpers

def _keys_dir(keys_dir: Path | str | None) -> Path:
    return Path(keys_dir) if keys_dir is not None else PATHS["keys"]


def _passphrase() -> bytes | None:
    value = os.environ.get(PASSPHRASE_ENV)
    return value.encode("utf-8") if value else None


def is_sha256_hex(value: Any) -> bool:
    """True if `value` is a 64-character lowercase hex string."""
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for hashing and for log lines."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def compute_record_hash(record: Mapping[str, Any]) -> str:
    """SHA-256 hex of the canonical JSON of all fields except record_hash and signature."""
    body = {key: value for key, value in record.items() if key not in _NOT_HASHED}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def file_sha256(path: Path | str) -> str:
    """SHA-256 hex of a file, read in 1 MB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _relative(path: Path, root: Path | str | None) -> str:
    if root is None:
        return rel_to_root(path)
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        raise ValueError(f"{path.name} is outside the project root") from None


def _normalize_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Round-trip params through JSON so the stored and hashed values are identical."""
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")
    return json.loads(json.dumps(dict(params), default=str, allow_nan=False))


# --------------------------------------------------------------------- keys

def _write_secret(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def _write_public(path: Path, public_key: Ed25519PublicKey) -> None:
    path.write_bytes(public_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    ))


def _load_public_file(path: Path) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(Path(path).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError(f"{Path(path).name} is not an Ed25519 public key")
    return key


def public_key_id(public_key: Ed25519PublicKey | None = None, *, keys_dir: Path | str | None = None) -> str:
    """First 16 hex characters of SHA-256 of the raw 32-byte public key."""
    if public_key is None:
        public_key = _load_public_file(_keys_dir(keys_dir) / PUBLIC_KEY_NAME)
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]


def load_private_key(keys_dir: Path | str | None = None) -> Ed25519PrivateKey:
    """Load the signing key (cached). Uses PROVENANCE_KEY_PASSPHRASE if set."""
    path = _keys_dir(keys_dir) / PRIVATE_KEY_NAME
    if not path.exists():
        raise FileNotFoundError("no private key; run: python -m src.trust.provenance --init-keys")
    data = path.read_bytes()
    password = _passphrase()
    cache_key = hashlib.sha256(data + (password or b"")).hexdigest()
    if cache_key not in _private_cache:
        key = serialization.load_pem_private_key(data, password=password)
        if not isinstance(key, Ed25519PrivateKey):
            raise TypeError("the private key is not an Ed25519 key")
        _private_cache[cache_key] = key
    return _private_cache[cache_key]


def load_public_keys(keys_dir: Path | str | None = None) -> dict[str, Ed25519PublicKey]:
    """All public keys in keys/ (current and archived), keyed by public_key_id."""
    keyring: dict[str, Ed25519PublicKey] = {}
    for path in sorted(_keys_dir(keys_dir).glob("*public*.pem")):
        try:
            key = _load_public_file(path)
        except (ValueError, TypeError):
            log.warning("skipping unreadable public key file %s", path.name)
            continue
        keyring[public_key_id(key)] = key
    return keyring


def _archive_public(pub_path: Path) -> None:
    """Rename the current public key to ed25519_public_<id>.pem so old records still verify."""
    try:
        old_id = public_key_id(_load_public_file(pub_path))
    except (ValueError, TypeError):
        log.warning("removing unreadable public key file %s", pub_path.name)
        pub_path.unlink()
        return
    archive = pub_path.with_name(f"ed25519_public_{old_id}.pem")
    if archive.exists():
        pub_path.unlink()
    else:
        pub_path.replace(archive)
    log.info("archived public key %s", old_id)


def init_keys(force: bool = False, *, keys_dir: Path | str | None = None) -> str:
    """Create keys/ed25519_private.pem (gitignored) and keys/ed25519_public.pem.

    Returns the public key id. With force=True a new key replaces the old one and
    the old public key is archived, so records it signed still verify.
    """
    kd = _keys_dir(keys_dir)
    kd.mkdir(parents=True, exist_ok=True)
    priv_path, pub_path = kd / PRIVATE_KEY_NAME, kd / PUBLIC_KEY_NAME

    if priv_path.exists() and not force:
        private = load_private_key(kd)
        key_id = public_key_id(private.public_key())
        if not pub_path.exists():
            _write_public(pub_path, private.public_key())
            log.info("restored missing public key file")
        elif public_key_id(_load_public_file(pub_path)) != key_id:
            log.warning("%s does not match the private key", PUBLIC_KEY_NAME)
        return key_id

    if pub_path.exists():
        _archive_public(pub_path)
    private = Ed25519PrivateKey.generate()
    password = _passphrase()
    encryption = (
        serialization.BestAvailableEncryption(password) if password
        else serialization.NoEncryption()
    )
    _write_secret(priv_path, private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, encryption,
    ))
    _write_public(pub_path, private.public_key())
    key_id = public_key_id(private.public_key())
    log.info("created signing key %s", key_id)
    return key_id


# ---------------------------------------------------------------------- log

def _last_line(path: Path) -> str | None:
    """Last non-empty line of a text file, read from the end."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        pos = fh.tell()
        tail = b""
        while pos > 0:
            step = min(4096, pos)
            pos -= step
            fh.seek(pos)
            tail = fh.read(step) + tail
            body = tail.rstrip(b"\r\n\t ")
            if b"\n" in body:
                return body.rsplit(b"\n", 1)[1].strip().decode("utf-8")
    body = tail.strip()
    return body.decode("utf-8") if body else None


def _needs_newline(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as fh:
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) != b"\n"


def _read_last_record(log_path: Path) -> dict[str, Any] | None:
    line = _last_line(log_path)
    if line is None:
        return None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        raise ValueError("the last log record is not valid JSON; run: python -m src.trust.verify") from None
    if not isinstance(record, dict) or "seq" not in record or not is_sha256_hex(record.get("record_hash")):
        raise ValueError("the last log record is malformed; run: python -m src.trust.verify")
    return record


def append_record(
    path: Path | str | None,
    kind: str,
    source_url: str | None = None,
    params: Mapping[str, Any] | None = None,
    *,
    sha256: str | None = None,
    log_path: Path | str | None = None,
    keys_dir: Path | str | None = None,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Hash `path`, chain to the last record, sign, and append one JSON line.

    path is stored relative to the project root. For records without a file
    (kind "reading"), pass path=None and the sha256 to record. Keys are created
    on first use.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if path is None:
        if sha256 is None:
            raise ValueError("give a file path or a sha256")
        if not is_sha256_hex(sha256):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        rel, digest = None, sha256
    else:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"not a file: {file_path.name}")
        rel = _relative(file_path, root)
        digest = file_sha256(file_path)
        if sha256 is not None and sha256 != digest:
            raise ValueError("the given sha256 does not match the file")

    log_file = Path(log_path) if log_path is not None else PATHS["provenance_log"]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    kd = _keys_dir(keys_dir)
    if not (kd / PRIVATE_KEY_NAME).exists():
        log.info("no signing key found; creating one")
        init_keys(keys_dir=kd)
    private = load_private_key(kd)

    last = _read_last_record(log_file)
    record: dict[str, Any] = {
        "seq": 1 if last is None else int(last["seq"]) + 1,
        "kind": kind,
        "path": rel,
        "source_url": source_url,
        "params": _normalize_params(params),
        "created_at": _utc_now(),
        "sha256": digest,
        "code_version": code_version(),
        "prev_hash": GENESIS_HASH if last is None else last["record_hash"],
        "public_key_id": public_key_id(private.public_key()),
    }
    record["record_hash"] = compute_record_hash(record)
    record["signature"] = base64.b64encode(
        private.sign(bytes.fromhex(record["record_hash"]))
    ).decode("ascii")

    prefix = "\n" if _needs_newline(log_file) else ""
    with open(log_file, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(prefix + canonical_json(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    log.debug("provenance record %d (%s) %s", record["seq"], kind, rel or "(no file)")
    return dict(record)


def read_log(log_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Parse the log into a list of records. Blank lines are skipped.

    Raises LogFormatError for a line that is not a JSON object.
    """
    path = Path(log_path) if log_path is not None else PATHS["provenance_log"]
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                position = len(records) + 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    raise LogFormatError(position, "not valid JSON") from None
                if not isinstance(obj, dict):
                    raise LogFormatError(position, "not a JSON object")
                records.append(obj)
    except UnicodeDecodeError:
        raise LogFormatError(len(records) + 1, "not UTF-8 text") from None
    return records


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m src.trust.provenance --init-keys [--force] | --key-id"""
    parser = argparse.ArgumentParser(description="Manage the provenance signing key.")
    parser.add_argument("--init-keys", action="store_true", help="create the key pair if missing")
    parser.add_argument("--force", action="store_true",
                        help="with --init-keys: replace the key (the old public key is archived)")
    parser.add_argument("--key-id", action="store_true", help="print the current public key id")
    args = parser.parse_args(argv)
    if not (args.init_keys or args.key_id):
        parser.print_help()
        return 0
    if args.init_keys:
        print(f"public key id: {init_keys(force=args.force)}")
    if args.key_id:
        print(public_key_id())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
