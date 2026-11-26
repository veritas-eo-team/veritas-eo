"""Local JSON API behind the VERITAS web console.

Standard library only, binds to 127.0.0.1. The Next.js console talks to this
process; every number it shows still comes from src.pipeline, so the web layer
cannot drift from the evaluated code.

    python scripts/api_server.py --port 8077
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import tempfile
import threading
import traceback
from collections import OrderedDict
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import PATHS, load_config  # noqa: E402

CLRSKY = "ctx_CLRSKY_SFC_SW_DWN"
ALARM_ACTIONS = ("verify", "quarantine")
DETECTOR_COLUMNS = (
    ("step", "step_score"),
    ("drift", "drift_score"),
    ("replay", "replay_score"),
    ("consistency", "consistency_score"),
)
UNITS = ("mj", "kwh", "wm2")
MAX_CACHED_RUNS = 8

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"cfg": None, "model": None, "model_error": None}
_RUNS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()


# ------------------------------------------------------------------ json helpers

def _num(value: Any, digits: int = 4) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return round(out, digits)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return None
    return stamp.strftime("%Y-%m-%d")


# ------------------------------------------------------------------ state

def cfg() -> dict:
    if _STATE["cfg"] is None:
        _STATE["cfg"] = load_config()
    return _STATE["cfg"]


def model() -> Any:
    """Trained context model, or None when the app must run in demo mode."""
    with _LOCK:
        if _STATE["model"] is None and _STATE["model_error"] is None:
            try:
                from src.detect.context_model import ContextModel
                _STATE["model"] = ContextModel.load()
            except Exception as exc:  # noqa: BLE001 - surfaced as demo mode
                _STATE["model_error"] = str(exc)
    return _STATE["model"]


def locations(split: str | None = None) -> list[str]:
    try:
        from src.data.align import clean_path, list_locations
        return [loc for loc in list_locations(split, cfg()) if clean_path(loc).exists()]
    except Exception:  # noqa: BLE001
        return []


_SYNTH: dict[str, Any] = {}


def synthetic() -> tuple[pd.DataFrame, Any]:
    """Deterministic stand-in when data or the model are missing."""
    from src.synthetic import make_synthetic_frame

    with _LOCK:
        if "frame" not in _SYNTH:
            from src.detect.context_model import ContextModel
            frames = [make_synthetic_frame(location=f"train{i}", seed=i) for i in range(3)]
            _SYNTH["model"] = ContextModel(cfg=cfg()).fit(frames)
            _SYNTH["frame"] = make_synthetic_frame(location="demo", seed=7, split="test")
        return _SYNTH["frame"], _SYNTH["model"]


def inputs(location: str) -> tuple[pd.DataFrame, Any, str, bool]:
    """(frame, model, label, demo_mode)."""
    trained = model()
    if trained is not None and location in locations():
        try:
            from src.data.align import load
            return load(location, cfg()), trained, f"{location} · NASA POWER", False
        except Exception:  # noqa: BLE001
            pass
    frame, fitted = synthetic()
    return frame, fitted, "synthetic demo location", True


# ------------------------------------------------------------------ scenarios

def attack_window(df: pd.DataFrame, attack: str, size: float | None, duration: int | None,
                  noise: float, start: str | None, seed: int) -> tuple[pd.DataFrame, dict]:
    """Inject on the full series, then cut [start - warm-up, end of test year]."""
    from src.attacks.inject import inject

    conf = cfg()
    ty = int(conf["test_year"])
    settings = dict(conf.get("eval") or {})
    window = settings.get("attack_window") or ["02-01", "09-30"]
    warmup = int(settings.get("warmup_days") or 365)
    kw: dict[str, Any] = {"seed": int(seed), "window_start": f"{ty}-{window[0]}",
                          "window_end": f"{ty}-{window[1]}"}
    if attack != "none":
        if size is not None:
            kw["size"] = float(size)
        if duration is not None:
            kw["duration"] = int(duration)
        if attack == "replay":
            kw["noise"] = float(noise)
    attacked, labels = inject(df["target"], attack, start=start, **kw)
    anchor = labels["start"] if labels["start"] is not None else pd.Timestamp(f"{ty}-01-01")
    begin = max(df.index[0], anchor - pd.Timedelta(days=warmup))
    run_df = df.loc[begin:f"{ty}-12-31"].copy()
    run_df["target"] = attacked.loc[run_df.index]
    return run_df, labels


def scenario(location: str, attack: str, size: float | None, duration: int | None,
             noise: float, start: str | None, seed: int) -> dict:
    from src.pipeline import run_pipeline

    frame, fitted, label, demo = inputs(location)
    run_df, labels = attack_window(frame, attack, size, duration, noise, start, seed)
    result = run_pipeline(run_df, fitted, cfg())
    payload = payload_for(result, run_df, labels, attack)
    payload.update(locationLabel=label, demo=demo)
    payload["runId"] = _remember(result, run_df, labels)
    return payload


def _remember(result: pd.DataFrame, run_df: pd.DataFrame, labels: dict) -> str:
    run_id = hashlib.sha1(f"{id(result)}-{len(_RUNS)}-{pd.Timestamp.now()}".encode()).hexdigest()[:12]
    with _LOCK:
        _RUNS[run_id] = {"result": result, "frame": run_df, "labels": labels}
        while len(_RUNS) > MAX_CACHED_RUNS:
            _RUNS.popitem(last=False)
    return run_id


# ------------------------------------------------------------------ payloads

def summarize(result: pd.DataFrame, labels: dict) -> dict:
    alarm = result["action"].isin(ALARM_ACTIONS)
    out: dict[str, Any] = {"trueStart": _iso(labels.get("start")),
                           "trueEnd": _iso(labels.get("end")),
                           "firstAlarm": None, "delay": None, "type": None, "estStart": None,
                           "lowestTrust": _num(result["trust"].min(), 2),
                           "alarmDays": int(alarm.sum())}
    start = labels.get("start")
    if start is None:
        return out
    after = result.index >= start
    out["lowestTrust"] = _num(result.loc[after, "trust"].min(), 2)
    hits = alarm[after]
    if hits.any():
        first = hits[hits].index[0]
        out.update(firstAlarm=_iso(first), delay=int((first - start).days),
                   type=str(result.loc[first, "attack_type"]),
                   estStart=_iso(result.loc[first, "est_start"]))
    return out


def series(result: pd.DataFrame, run_df: pd.DataFrame) -> dict:
    clear = (run_df[CLRSKY].reindex(result.index) if CLRSKY in run_df
             else pd.Series(np.nan, index=result.index))
    gap_pct = (np.exp(result["gap"].to_numpy(dtype=float)) - 1.0) * 100.0
    payload = {
        "dates": [d.strftime("%Y-%m-%d") for d in result.index],
        "observed": [_num(v, 3) for v in result["target"]],
        "expected": [_num(v, 3) for v in result["expected"]],
        "clearSky": [_num(v, 3) for v in clear],
        "gapPct": [_num(v, 2) for v in gap_pct],
        "trust": [_num(v, 3) for v in result["trust"]],
        "action": [str(a) for a in result["action"]],
        "attackType": [str(t) for t in result["attack_type"]],
        "alert": [bool(a) for a in result["alert"]],
        "scores": {name: [_num(v, 3) for v in result[col].clip(upper=3.0)]
                   for name, col in DETECTOR_COLUMNS},
    }
    return payload


def payload_for(result: pd.DataFrame, run_df: pd.DataFrame, labels: dict, attack: str) -> dict:
    flagged = result[result["action"] != "accept"]
    rows = [{
        "date": _iso(day),
        "action": str(flagged.loc[day, "action"]),
        "trust": _num(flagged.loc[day, "trust"], 2),
        "gapPct": _num((np.exp(float(flagged.loc[day, "gap"])) - 1.0) * 100.0, 1),
        "attackType": str(flagged.loc[day, "attack_type"]),
        "estStart": _iso(flagged.loc[day, "est_start"]),
    } for day in flagged.index]
    return {
        "attack": attack,
        "labels": {"start": _iso(labels.get("start")), "end": _iso(labels.get("end")),
                   "type": labels.get("type")},
        "summary": summarize(result, labels),
        "series": series(result, run_df),
        "flagged": rows,
        "days": int(len(result)),
    }


def explain(run_id: str, day: str) -> dict:
    """Plain-language reasons for one day, strongest first."""
    run = _RUNS.get(run_id)
    if run is None:
        raise KeyError("that run is no longer in memory; run the scenario again")
    result, run_df = run["result"], run["frame"]
    stamp = pd.Timestamp(day)
    if stamp not in result.index:
        raise KeyError(f"{day} is not part of this run")
    row = result.loc[stamp]
    slope = float((cfg().get("fusion") or {}).get("slope") or 8.0)
    reasons: list[list[str]] = []
    target = float(row["target"])
    gap_pct = float((np.exp(float(row["gap"])) - 1.0) * 100.0) if pd.notna(row["gap"]) else float("nan")

    if pd.notna(row["z"]):
        z = float(row["z"])
        level = "strong" if abs(z) >= 3 else "medium" if abs(z) >= 2 else "info"
        reasons.append([level, (
            f"The site reports {target:.2f} MJ/m²/day, {gap_pct:+.1f}% away from the "
            f"{float(row['expected']):.2f} expected from neighbours, clear-sky radiation and "
            f"weather ({z:+.1f} standard deviations).")])
    nb_cols = [c for c in run_df.columns if c.startswith("nb_")]
    if nb_cols:
        nb = run_df.loc[stamp, nb_cols].astype(float).dropna()
        if len(nb):
            reasons.append(["info", (
                "Neighbouring grid cells read "
                + ", ".join(f"{v:.2f}" for v in nb)
                + f" MJ/m²/day (mean {nb.mean():.2f}).")])
    clr = (float(run_df.loc[stamp, CLRSKY])
           if CLRSKY in run_df and pd.notna(run_df.loc[stamp, CLRSKY]) else None)
    clearness = target / clr if clr and clr > 0 else None
    if clr is not None:
        if target > clr:
            reasons.append(["strong", (
                f"The value is above the clear-sky limit ({clr:.2f}): more sunshine than a "
                "cloudless sky can deliver, which is physically impossible.")])
        else:
            reasons.append(["info", f"Clear-sky limit {clr:.2f}; the reading is {clearness:.0%} of it."])
    if "cloud" in run_df and pd.notna(run_df.loc[stamp, "cloud"]):
        cloud = float(run_df.loc[stamp, "cloud"])
        if cloud >= 0.7 and clearness is not None and clearness >= 0.8:
            reasons.append(["medium", (
                f"Sentinel-2 saw {cloud:.0%} cloud over the site today, yet the reading is "
                f"{clearness:.0%} of clear-sky (weak evidence alone: one 10:30 snapshot).")])
        else:
            reasons.append(["info", f"Sentinel-2 passed over today and saw {cloud:.0%} cloud."])
    titles = {"step": ("Step (CUSUM)", "a sustained shift has built up in the gap"),
              "drift": ("Drift (EWMA + slope)", "the gap is trending slowly away from zero"),
              "replay": ("Replay / stuck", "these days repeat earlier data or a frozen value"),
              "consistency": ("Physics checks", "repeated clear-sky or cloud contradictions")}
    contrib = []
    for name, col in DETECTOR_COLUMNS:
        score = float(row[col])
        title, meaning = titles[name]
        contrib.append({"detector": title, "score": _num(score, 3),
                        "p": _num(1.0 / (1.0 + float(np.exp(-np.clip(slope * (score - 1.0), -50, 50)))), 3)})
        if score >= 1.0:
            reasons.append(["strong", f"{title} is above its alarm level (score {score:.2f}): {meaning}."])
        elif score >= 0.75:
            reasons.append(["medium", f"{title} is close to its alarm level (score {score:.2f}): {meaning}."])
    if pd.notna(row["est_start"]):
        reasons.append(["info", f"Estimated start of the manipulation: {_iso(row['est_start'])}."])

    order = {"strong": 0, "medium": 1, "info": 2}
    reasons.sort(key=lambda r: order[r[0]])
    return {"date": _iso(stamp), "action": str(row["action"]), "trust": _num(row["trust"], 2),
            "p": _num(row["p_compromise"], 3), "type": str(row["attack_type"]),
            "gapPct": _num(gap_pct, 1), "reasons": reasons, "contrib": contrib}


# ------------------------------------------------------------------ meta, benchmark

def benchmark() -> dict:
    path = PATHS["results"] / "benchmark_table.csv"
    if not path.exists():
        return {"rows": [], "summary": None, "images": []}
    table = pd.read_csv(path)
    rows = [{key: (int(value) if isinstance(value, (int, np.integer))
                              else _num(value, 4) if isinstance(value, (float, np.floating))
                              else str(value))
             for key, value in record.items()} for record in table.to_dict("records")]
    summary_path = PATHS["results"] / "summary.md"
    charts = [("catch_vs_size.png", "Catch rate vs step size"),
              ("delay_vs_drift_duration.png", "Delay vs drift duration"),
              ("false_alarms_vs_threshold.png", "False alarms vs threshold"),
              ("reliability.png", "Reliability of the compromise probability"),
              ("attack_examples.png", "The attacks")]
    images = [{"file": name, "title": title}
              for name, title in charts if (PATHS["results"] / name).exists()]
    return {
        "rows": rows,
        "summary": summary_path.read_text(encoding="utf-8") if summary_path.exists() else None,
        "images": images,
    }


def meta() -> dict:
    configured = bool(PATHS["tuned"].exists())
    trained = model() is not None
    train, test = locations("train"), locations("test")
    log = provenance_status()
    return {
        "testYear": int(cfg()["test_year"]),
        "cleanLocations": len(locations()),
        "train": train,
        "test": test,
        "model": trained,
        "modelError": _STATE["model_error"],
        "tuned": configured,
        "provenance": log,
        "benchmark": benchmark(),
        "units": "MJ/m²/day",
        "warmupDays": int((cfg().get("eval") or {}).get("warmup_days") or 365),
        "attackWindow": (cfg().get("eval") or {}).get("attack_window") or ["02-01", "09-30"],
    }


# ------------------------------------------------------------------ provenance

def provenance_status() -> dict:
    try:
        from src.trust.verify import verify_log
        res = verify_log(check_files=False)
        return {"ok": bool(res.ok), "records": int(res.n_records), "reason": res.reason,
                "head": str(res.head_hash)[:16] if getattr(res, "head_hash", None) else None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "records": 0, "reason": str(exc), "head": None}


def record_checks(records, keys) -> list[bool]:
    from src.trust.provenance import GENESIS_HASH, compute_record_hash

    out, prev = [], GENESIS_HASH
    for rec in records:
        ok = compute_record_hash(rec) == rec.get("record_hash") and rec.get("prev_hash") == prev
        key = keys.get(rec.get("public_key_id"))
        try:
            key.verify(base64.b64decode(rec["signature"]), bytes.fromhex(rec["record_hash"]))
        except Exception:  # noqa: BLE001
            ok = False
        out.append(bool(ok))
        prev = rec.get("record_hash")
    return out


def integrity() -> dict:
    from src.trust.provenance import load_public_keys, read_log

    try:
        records = read_log()
    except Exception as exc:  # noqa: BLE001
        return {"records": [], "keys": [], "error": str(exc), "verified": provenance_status()}
    keys = load_public_keys()
    checks = record_checks(records, keys) if records else []
    rows = [{
        "seq": rec.get("seq"), "kind": rec.get("kind"), "path": rec.get("path") or "—",
        "created": rec.get("created_at"), "fileHash": str(rec.get("sha256", ""))[:12],
        "recordHash": str(rec.get("record_hash", ""))[:12], "keyId": rec.get("public_key_id"),
        "ok": bool(ok),
    } for rec, ok in zip(records, checks)]
    return {"records": rows, "keys": sorted(keys), "verified": provenance_status()}


def integrity_action(action: str) -> dict:
    if action == "verify":
        from src.trust.verify import verify_log
        res = verify_log(check_files=True, allow_missing=True)
        return {"ok": bool(res.ok), "records": int(res.n_records), "reason": res.reason,
                "firstBad": getattr(res, "first_bad_seq", None), "head": str(res.head_hash)}
    if action == "tamper":
        from src.trust.verify import tamper_demo
        res = tamper_demo()
        return {"caught": not bool(res.ok), "reason": res.reason,
                "firstBad": getattr(res, "first_bad_seq", None)}
    if action == "gateway":
        from src.trust.gateway import demo_attacks
        cases = demo_attacks()[["case", "expected", "accepted", "reason"]]
        return {"cases": [{"case": str(r["case"]), "expected": str(r["expected"]),
                           "accepted": bool(r["accepted"]), "reason": str(r["reason"])}
                          for r in cases.to_dict("records")]}
    if action == "overhead":
        from src.trust.gateway import bench_gateway
        from src.trust.verify import bench_overhead
        b = bench_overhead(n=300)
        g = bench_gateway(n=100)
        return {"rows": [
            {"operation": "sign one provenance record", "value": f"{b['sign_ms']:.3f} ms"},
            {"operation": "verify one record (chain)", "value": f"{b['chain_verify_ms_per_record']:.3f} ms"},
            {"operation": "hash 1 MB", "value": f"{b['hash_ms_per_mb']:.3f} ms"},
            {"operation": "provenance record size", "value": f"{b['bytes_per_record']:.0f} bytes"},
            {"operation": "sign one reading (sensor)", "value": f"{g['sign_ms']:.3f} ms"},
            {"operation": "check one reading (gateway)", "value": f"{g['check_ms']:.3f} ms"},
            {"operation": "check + signed log append", "value": f"{g['check_and_log_ms']:.3f} ms"},
        ]}
    raise KeyError(f"unknown integrity action {action!r}")


# ------------------------------------------------------------------ upload

def score_csv(body: dict) -> dict:
    from src import score_file as SF
    from src.data import power

    text = str(body.get("csv") or "")
    if not text.strip():
        raise SF.InputError("the uploaded file is empty")
    units = str(body.get("units") or "mj")
    if units not in UNITS:
        raise SF.InputError(f"units must be one of {list(UNITS)}")
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        target, extras = SF.read_input(path, (body.get("dateCol") or "").strip() or None,
                                       (body.get("valueCol") or "").strip() or None, units)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    warning = target.attrs.get("unit_warning")
    result, info = SF.score_series(target, cfg(), model(), extras=extras,
                                   lat=float(body.get("lat") or 0.0), lon=float(body.get("lon") or 0.0),
                                   name=str(body.get("name") or "uploaded series"))
    frame = info.pop("frame", None)
    labels = {"start": None, "end": None}
    known = body.get("knownStart")
    if known:
        labels = {"start": pd.Timestamp(known), "end": result.index[-1]}
    payload = payload_for(result, frame if frame is not None else pd.DataFrame(index=result.index),
                          labels, "upload")
    episodes = SF.alarm_episodes(result)
    payload.update(
        name=str(info.get("name") or "uploaded series"),
        demo=False,
        warning=warning,
        info={key: (str(value) if isinstance(value, (pd.Timestamp, date)) else value)
              for key, value in info.items() if key not in {"neighbours", "frame"}},
        neighbours=list(info.get("neighbours") or []),
        episodes=[{"from": _iso(e["start"]), "to": _iso(e["end"]), "days": e["days"],
                   "type": e["type"], "estStart": _iso(e["est_start"])} for e in episodes],
        firstAlarm=_iso(pd.Timestamp(result.index[result["action"].isin(ALARM_ACTIONS)][0]))
        if result["action"].isin(ALARM_ACTIONS).any() else None,
        trustMin=_num(result["trust"].min(), 2),
        quarantined=int((result["action"] == "quarantine").sum()),
        csv=result.to_csv(date_format="%Y-%m-%d", index_label="date"),
    )
    payload["runId"] = _remember(result, frame if frame is not None else pd.DataFrame(index=result.index),
                                labels)
    return payload


def demo_csv() -> dict:
    frame = None
    try:
        from src.data.align import load
        frame = load("kairouan", cfg()).loc["2023-01-01":"2024-12-31"].copy()
    except Exception:  # noqa: BLE001
        base, _ = synthetic()
        frame = base.loc["2023-01-01":"2024-12-31"].copy()
    target = frame["target"].copy()
    target.loc["2024-05-01":] *= 1.12
    out = frame[[c for c in frame.columns if c.startswith(("nb_", "ctx_"))]].copy()
    out.insert(0, "value", target.to_numpy())
    out.insert(0, "date", frame.index.strftime("%Y-%m-%d"))
    return {"csv": out.to_csv(index=False), "name": "demo_kairouan_attacked",
            "lat": 35.68, "lon": 10.10, "units": "mj", "knownStart": "2024-05-01"}


def results_file(name: str) -> tuple[bytes, str]:
    allowed = {"catch_vs_size.png", "delay_vs_drift_duration.png",
               "false_alarms_vs_threshold.png", "reliability.png", "attack_examples.png"}
    if name not in allowed:
        raise KeyError("unknown results image")
    path = PATHS["results"] / name
    if not path.exists():
        raise FileNotFoundError(name)
    return path.read_bytes(), "image/png"


# ------------------------------------------------------------------ http

class Handler(BaseHTTPRequestHandler):
    server_version = "VeritasAPI/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
        sys.stderr.write("[api] " + (fmt % args) + "\n")

    # -- plumbing
    def _send(self, body: bytes, status: int = 200, kind: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, allow_nan=False, default=str).encode("utf-8")
        self._send(body, status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            raise ValueError("request body must be JSON") from None
        return data if isinstance(data, dict) else {}

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        try:
            if parsed.path == "/health":
                return self._json({"ok": True, "model": model() is not None})
            if parsed.path == "/api/meta":
                return self._json(meta())
            if parsed.path == "/api/integrity":
                return self._json(integrity())
            if parsed.path == "/api/explain":
                return self._json(explain(query.get("run", ""), query.get("day", "")))
            if parsed.path == "/api/demo-csv":
                return self._json(demo_csv())
            if parsed.path.startswith("/api/results/"):
                data, kind = results_file(parsed.path.rsplit("/", 1)[-1])
                return self._send(data, 200, kind)
            if parsed.path == "/api/scenario":
                args = dict(location=query.get("location", "kairouan"),
                            attack=query.get("attack", "step"),
                            size=float(query["size"]) if query.get("size") else None,
                            duration=int(query["duration"]) if query.get("duration") else None,
                            noise=float(query.get("noise") or 0.0),
                            start=query.get("start") or None,
                            seed=int(query.get("seed") or 1))
                return self._json(scenario(**args))
            return self._json({"error": f"no route {parsed.path}"}, 404)
        except KeyError as exc:
            return self._json({"error": str(exc.args[0])}, 404)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        except FileNotFoundError as exc:
            return self._json({"error": f"missing file: {exc}"}, 404)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urlparse(self.path)
        try:
            body = self._body()
            if parsed.path == "/api/scenario":
                args = dict(location=str(body.get("location") or "kairouan"),
                            attack=str(body.get("attack") or "step"),
                            size=float(body["size"]) if body.get("size") is not None else None,
                            duration=int(body["duration"]) if body.get("duration") is not None else None,
                            noise=float(body.get("noise") or 0.0),
                            start=body.get("start") or None,
                            seed=int(body.get("seed") or 1))
                return self._json(scenario(**args))
            if parsed.path.startswith("/api/integrity/"):
                return self._json(integrity_action(parsed.path.rsplit("/", 1)[-1]))
            if parsed.path == "/api/score":
                return self._json(score_csv(body))
            return self._json({"error": f"no route {parsed.path}"}, 404)
        except KeyError as exc:
            return self._json({"error": str(exc.args[0])}, 404)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            from src import score_file as SF
            from src.data import power
            if isinstance(exc, (SF.InputError, power.PowerAPIError, FileNotFoundError)):
                return self._json({"error": str(exc)}, 400)
            traceback.print_exc()
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main() -> int:
    parser = argparse.ArgumentParser(description="VERITAS local JSON API")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"VERITAS API on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
