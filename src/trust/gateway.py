"""Emulated signed sensor and gateway: a software trust anchor for individual readings.

Sensor(device_id, private_key).sign_reading(date, value) -> message
    {device_id, date, value, seq, ts, nonce, sig}; seq goes up by 1 per reading; sig is an
    Ed25519 signature (base64) over the canonical JSON of the other fields.
Gateway(public_keys, ts_window_s=300).accept(msg) -> (accepted, reason). Checks, in order:
    complete message, known device, valid signature, timestamp within the window,
    sequence strictly increasing, nonce not used before. Accepted readings are appended to
    data/provenance/readings.jsonl with the same signed hash-chain format as the
    provenance log (kind "reading", sha256 of the full message).

What this proves and what it cannot: altered, replayed, delayed or forged messages are
rejected, but a compromised sensor holding a valid key can send biased values that pass
every check. That is why the statistical detectors exist.

CLI: python -m src.trust.gateway --demo | --bench   (temporary keys and log; real data untouched)
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import math
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from src.common import PATHS, get_logger
from src.trust.provenance import append_record, canonical_json

log = get_logger(__name__)

MESSAGE_FIELDS: tuple[str, ...] = ("device_id", "date", "value", "seq", "ts", "nonce", "sig")
SIGNED_FIELDS: tuple[str, ...] = MESSAGE_FIELDS[:-1]
DEFAULT_WINDOW_S = 300.0


def _payload(msg: Mapping[str, Any]) -> bytes:
    return canonical_json({k: msg[k] for k in SIGNED_FIELDS}).encode("utf-8")


def message_digest(msg: Mapping[str, Any]) -> str:
    """SHA-256 hex of the canonical JSON of the full message (signature included)."""
    return hashlib.sha256(canonical_json(dict(msg)).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ sensor

class Sensor:
    """Emulated field sensor that signs each daily reading with its own Ed25519 key."""

    def __init__(self, device_id: str, private_key: Ed25519PrivateKey, *,
                 clock: Callable[[], float] = time.time,
                 nonce_fn: Callable[[], str] | None = None, start_seq: int = 0) -> None:
        self.device_id = str(device_id)
        self._key = private_key
        self.clock = clock
        self.nonce_fn = nonce_fn or (lambda: secrets.token_hex(16))
        self.seq = int(start_seq)

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._key.public_key()

    def __repr__(self) -> str:
        return f"Sensor({self.device_id!r}, seq={self.seq})"

    def sign_reading(self, date: Any, value: float) -> dict[str, Any]:
        """Signed message for one reading; the sequence number goes up by one."""
        v = float(value)
        if not math.isfinite(v):
            raise ValueError("a reading must be a finite number")
        self.seq += 1
        msg: dict[str, Any] = {
            "device_id": self.device_id,
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "value": round(v, 6),
            "seq": self.seq,
            "ts": round(float(self.clock()), 3),
            "nonce": str(self.nonce_fn()),
        }
        msg["sig"] = base64.b64encode(self._key.sign(_payload(msg))).decode("ascii")
        return msg


# ------------------------------------------------------------------ gateway

class Gateway:
    """Checks signed readings and logs the accepted ones in a signed hash chain."""

    def __init__(self, public_keys: Mapping[str, Ed25519PublicKey], ts_window_s: float = DEFAULT_WINDOW_S, *,
                 log_path: Path | str | None = None, keys_dir: Path | str | None = None,
                 clock: Callable[[], float] = time.time, write_log: bool = True) -> None:
        self.public_keys = dict(public_keys)
        self.window = float(ts_window_s)
        self.log_path = Path(log_path) if log_path is not None else PATHS["readings_log"]
        self.keys_dir = keys_dir
        self.clock = clock
        self.write_log = write_log
        self.last_seq: dict[str, int] = {}
        self._nonces: dict[str, float] = {}

    def _prune(self, now: float) -> None:
        """Forget nonces old enough that the timestamp check already rejects them."""
        horizon = now - 2.0 * self.window
        for key in [k for k, ts in self._nonces.items() if ts < horizon]:
            del self._nonces[key]

    def accept(self, msg: Any) -> tuple[bool, str]:
        """(accepted, reason) for one message."""
        if not isinstance(msg, Mapping) or any(k not in msg for k in MESSAGE_FIELDS):
            return False, "malformed message"
        device = str(msg["device_id"])
        key = self.public_keys.get(device)
        if key is None:
            return False, "unknown device"
        try:
            key.verify(base64.b64decode(msg["sig"], validate=True), _payload(msg))
        except (InvalidSignature, ValueError, TypeError):
            return False, "bad signature"
        now = float(self.clock())
        try:
            ts, seq = float(msg["ts"]), int(msg["seq"])
        except (TypeError, ValueError):
            return False, "malformed message"
        if abs(now - ts) > self.window:
            return False, "timestamp outside window"
        if seq <= self.last_seq.get(device, 0):
            return False, "sequence not increasing"
        self._prune(now)
        nonce_key = f"{device}:{msg['nonce']}"
        if nonce_key in self._nonces:
            return False, "nonce reused"
        self.last_seq[device] = seq
        self._nonces[nonce_key] = ts
        if self.write_log:
            params = {k: msg[k] for k in SIGNED_FIELDS}
            params["device_sig"] = msg["sig"]
            append_record(None, "reading", None, params, sha256=message_digest(msg),
                          log_path=self.log_path, keys_dir=self.keys_dir)
        return True, "accepted"


# ------------------------------------------------------------------ streaming

def stream_series(series: pd.Series, sensor: Sensor, gateway: Gateway,
                  tamper: Callable[[pd.Timestamp, dict], dict | None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Send each non-NaN value through sensor -> (optional tamper in transit) -> gateway.

    Returns (accepted DataFrame[value, accepted] on series.index, rejections DataFrame[date, seq, reason]).
    `tamper(date, msg)` may modify the message or return None to drop it.
    """
    values = pd.Series(float("nan"), index=series.index, dtype="float64")
    accepted = pd.Series(False, index=series.index)
    rejections = []
    for date, value in series.items():
        if pd.isna(value):
            continue
        msg = sensor.sign_reading(date, float(value))
        if tamper is not None:
            msg = tamper(pd.Timestamp(date), dict(msg))
            if msg is None:
                rejections.append({"date": pd.Timestamp(date), "seq": sensor.seq, "reason": "dropped in transit"})
                continue
        ok, reason = gateway.accept(msg)
        if ok:
            values[date] = float(msg["value"])
            accepted[date] = True
        else:
            rejections.append({"date": pd.Timestamp(date), "seq": msg.get("seq"), "reason": reason})
    out = pd.DataFrame({"value": values, "accepted": accepted}, index=series.index)
    return out, pd.DataFrame(rejections, columns=["date", "seq", "reason"])


# ------------------------------------------------------------------ demo and overhead

def demo_attacks(workdir: Path | str | None = None) -> pd.DataFrame:
    """Run the attack cases against a fresh gateway in a temporary folder.

    Returns DataFrame[case, expected, accepted, reason, as_expected]; attrs log_ok tells
    whether the gateway's readings log verifies.
    """
    from src.trust.verify import verify_log

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(workdir) if workdir is not None else Path(tmp)
        log_path, keys_dir = base / "readings.jsonl", base / "keys"
        key = Ed25519PrivateKey.generate()
        sensor = Sensor("station-1", key)
        gw = Gateway({"station-1": key.public_key()}, log_path=log_path, keys_dir=keys_dir)
        rows = []

        def case(name: str, msg: Mapping[str, Any], expected: bool) -> None:
            ok, reason = gw.accept(msg)
            rows.append({"case": name, "expected": "accepted" if expected else "rejected",
                         "accepted": ok, "reason": reason, "as_expected": ok == expected})

        first = sensor.sign_reading("2024-05-01", 6.20)
        case("normal signed reading", first, True)
        altered = sensor.sign_reading("2024-05-02", 6.10)
        altered["value"] = round(altered["value"] * 1.12, 6)
        case("value altered in transit (+12%)", altered, False)
        case("old message replayed", dict(first), False)
        rogue = Sensor("station-1", Ed25519PrivateKey.generate(), start_seq=1000)
        case("forged with an unknown key", rogue.sign_reading("2024-05-02", 6.10), False)
        stranger = Sensor("station-9", Ed25519PrivateKey.generate())
        case("unregistered device", stranger.sign_reading("2024-05-02", 6.10), False)
        late = Sensor("station-1", key, clock=lambda: time.time() - 3600.0, start_seq=sensor.seq + 10)
        case("delayed message (1 h old)", late.sign_reading("2024-05-02", 6.10), False)
        case("compromised sensor, valid key, +12% bias", sensor.sign_reading("2024-05-03", 6.00 * 1.12), True)

        result = pd.DataFrame(rows)
        result.attrs["log_ok"] = bool(verify_log(log_path, check_files=False, keys_dir=keys_dir).ok)
        result.attrs["accepted_records"] = int(sum(r["accepted"] for r in rows))
    return result


def bench_gateway(n: int = 200) -> dict[str, float]:
    """Milliseconds per sign, per gateway check, and per check + signed log append."""
    with tempfile.TemporaryDirectory() as tmp:
        key = Ed25519PrivateKey.generate()
        sensor = Sensor("bench", key)
        msgs = [sensor.sign_reading("2024-01-01", 5.0 + i * 1e-3) for i in range(n)]
        t0 = time.perf_counter()
        for i in range(n):
            Sensor("bench", key, start_seq=i).sign_reading("2024-01-01", 5.0)
        sign_ms = (time.perf_counter() - t0) * 1000.0 / n
        gw = Gateway({"bench": key.public_key()}, write_log=False)
        t0 = time.perf_counter()
        for m in msgs:
            gw.accept(m)
        check_ms = (time.perf_counter() - t0) * 1000.0 / n
        gw_log = Gateway({"bench": key.public_key()}, log_path=Path(tmp) / "r.jsonl", keys_dir=Path(tmp) / "keys")
        t0 = time.perf_counter()
        for m in msgs:
            gw_log.accept(m)
        logged_ms = (time.perf_counter() - t0) * 1000.0 / n
        size = (Path(tmp) / "r.jsonl").stat().st_size / n
    return {"n": n, "sign_ms": sign_ms, "check_ms": check_ms, "check_and_log_ms": logged_ms,
            "bytes_per_logged_reading": size}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emulated signed sensor and gateway.")
    parser.add_argument("--demo", action="store_true", help="run the attack cases (temporary keys and log)")
    parser.add_argument("--bench", action="store_true", help="measure signing and checking overhead")
    args = parser.parse_args(argv)
    if not (args.demo or args.bench):
        parser.print_help()
        return 0
    code = 0
    if args.demo:
        result = demo_attacks()
        print(result[["case", "expected", "accepted", "reason"]].to_string(index=False))
        print(f"\nreadings log verifies: {result.attrs['log_ok']} ({result.attrs['accepted_records']} accepted readings)")
        print("Signatures stop altered, replayed, delayed and forged messages, but the compromised sensor's\n"
              "biased reading is ACCEPTED: only the statistical detectors can catch it.")
        code = 0 if result["as_expected"].all() and result.attrs["log_ok"] else 1
    if args.bench:
        b = bench_gateway()
        print(f"\nper reading: sign {b['sign_ms']:.3f} ms, check {b['check_ms']:.3f} ms, "
              f"check + signed log append {b['check_and_log_ms']:.3f} ms, "
              f"{b['bytes_per_logged_reading']:.0f} bytes logged (n = {b['n']})")
    return code


if __name__ == "__main__":
    raise SystemExit(main())