"""NASA POWER daily point downloader: a site cell plus 4 neighbour cells per location.

Endpoint: https://power.larc.nasa.gov/api/temporal/daily/point (community AG, JSON).
A single-point request accepts at most 20 parameters, so longer lists are batched.
Responses are cached to data/raw/power/<loc>/<cell_id>.json, never fetched twice
unless --force, and each new file is recorded in the provenance log.

CLI: python -m src.data.power --all | --location tunis [--force]
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Mapping

import pandas as pd
import requests

from src.common import PATHS, get_location, get_logger, load_config, rel_to_root
from src.trust.provenance import append_record

log = get_logger(__name__)

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
COMMUNITY = "AG"
MAX_PARAMS_PER_REQUEST = 20
FILL_THRESHOLD = -990.0
REQUEST_DELAY_S = 1.0
TIMEOUT_S = 60.0
RETRIES = 3
BACKOFF_S = 2.0
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
SITE_CELL = "site"
NEIGHBOR_CELLS: dict[str, tuple[float, float]] = {
    "nb_N": (1.0, 0.0),
    "nb_E": (0.0, 1.0),
    "nb_S": (-1.0, 0.0),
    "nb_W": (0.0, -1.0),
}
USER_AGENT = "veritas-eo/0.1 (research prototype)"

_sleep = time.sleep
_last_request: float = 0.0


class PowerAPIError(RuntimeError):
    """The POWER API returned an error or an unusable response."""


# ------------------------------------------------------------- requests

def _ymd(value: Any) -> str:
    return pd.Timestamp(str(value)).strftime("%Y%m%d")


def _wrap_lon(lon: float) -> float:
    return round(((lon + 180.0) % 360.0) - 180.0, 4)


def cell_specs(location: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Site cell (target + context vars) and N/E/S/W neighbour cells (target only)."""
    lat, lon = float(location["lat"]), float(location["lon"])
    step = float(cfg["neighbor_offset_deg"])
    target = cfg["target"]
    site_params = [target] + [var for var in cfg["context_vars"] if var != target]
    specs = [{"cell_id": SITE_CELL, "lat": round(lat, 4), "lon": round(lon, 4), "parameters": site_params}]
    for cell_id, (dlat, dlon) in NEIGHBOR_CELLS.items():
        specs.append({
            "cell_id": cell_id,
            "lat": round(min(90.0, max(-90.0, lat + dlat * step)), 4),
            "lon": _wrap_lon(lon + dlon * step),
            "parameters": [target],
        })
    return specs


def build_request_params(lat: float, lon: float, parameters: list[str], start: Any, end: Any) -> dict[str, str]:
    """Query parameters for one daily point request."""
    return {
        "parameters": ",".join(parameters),
        "community": COMMUNITY,
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start": _ymd(start),
        "end": _ymd(end),
        "format": "JSON",
    }


def request_url(params: Mapping[str, str]) -> str:
    """Full request URL (stored in provenance records)."""
    return requests.Request("GET", BASE_URL, params=dict(params)).prepare().url


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def _throttle(delay_s: float) -> None:
    """Keep at least delay_s seconds between requests."""
    global _last_request
    wait = _last_request + delay_s - time.monotonic()
    if wait > 0:
        _sleep(wait)
    _last_request = time.monotonic()


def fetch_json(
    params: Mapping[str, str],
    *,
    session: Any = None,
    retries: int = RETRIES,
    backoff_s: float = BACKOFF_S,
    timeout_s: float = TIMEOUT_S,
    delay_s: float = REQUEST_DELAY_S,
) -> dict[str, Any]:
    """GET one request with throttling and exponential-backoff retries.

    Retries network errors, invalid JSON and HTTP 429/5xx up to `retries` times
    (waits backoff_s, 2*backoff_s, 4*backoff_s). Other HTTP errors fail at once.
    """
    session = session if session is not None else _make_session()
    last_error = "no attempt made"
    for attempt in range(retries + 1):
        if attempt:
            _sleep(backoff_s * 2 ** (attempt - 1))
        _throttle(delay_s)
        try:
            response = session.get(BASE_URL, params=dict(params), timeout=timeout_s)
        except requests.RequestException as exc:
            last_error = type(exc).__name__
            log.warning("request failed (%s), attempt %d of %d", last_error, attempt + 1, retries + 1)
            continue
        if response.status_code == 200:
            try:
                data = response.json()
            except ValueError:
                last_error = "response is not JSON"
                log.warning("%s, attempt %d of %d", last_error, attempt + 1, retries + 1)
                continue
            if not isinstance(data, dict):
                raise PowerAPIError("unexpected JSON response (not an object)")
            return data
        snippet = (response.text or "")[:300].replace("\n", " ")
        if response.status_code in RETRY_STATUS:
            last_error = f"HTTP {response.status_code}"
            log.warning("%s, attempt %d of %d", last_error, attempt + 1, retries + 1)
            continue
        raise PowerAPIError(f"HTTP {response.status_code}: {snippet}")
    raise PowerAPIError(f"gave up after {retries + 1} attempts: {last_error}")


def merge_responses(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge batched responses for the same point into one response."""
    merged = json.loads(json.dumps(parts[0]))
    for part in parts[1:]:
        merged.setdefault("properties", {}).setdefault("parameter", {}).update(
            part.get("properties", {}).get("parameter", {}))
        merged.setdefault("parameters", {}).update(part.get("parameters", {}))
    return merged


def _check_response(data: Mapping[str, Any], parameters: list[str]) -> None:
    values = data.get("properties", {}).get("parameter") if isinstance(data.get("properties"), dict) else None
    if not isinstance(values, dict):
        raise PowerAPIError("response has no properties.parameter")
    missing = [p for p in parameters if p not in values]
    if missing:
        raise PowerAPIError(f"response is missing parameters: {missing}")
    if not any(values[p] for p in parameters):
        raise PowerAPIError("response contains no values")
    for message in data.get("messages", []) or []:
        log.info("POWER message: %s", message)


# ------------------------------------------------------------- parsing

def parse_power_json(obj: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Daily point response -> (DataFrame, units per variable).

    DataFrame: daily DatetimeIndex named "date", one float column per variable.
    Values <= -990 (the fill value is -999) and the header fill_value become NaN.
    df.attrs holds lat, lon and elevation when the response has them.
    """
    try:
        values = obj["properties"]["parameter"]
    except (KeyError, TypeError):
        raise ValueError("not a NASA POWER daily point response (no properties.parameter)") from None
    if not isinstance(values, dict) or not values:
        raise ValueError("NASA POWER response has no parameters")

    columns: dict[str, pd.Series] = {}
    for var, series in values.items():
        if not isinstance(series, dict):
            raise ValueError(f"parameter {var} is not a date -> value mapping")
        columns[var] = pd.Series(series, dtype="float64")
    frame = pd.DataFrame(columns)
    frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d")
    frame.index.name = "date"
    frame = frame.sort_index().astype("float64")

    frame = frame.where(frame > FILL_THRESHOLD)
    fill = (obj.get("header") or {}).get("fill_value")
    if isinstance(fill, (int, float)):
        frame = frame.mask(frame == float(fill))

    meta = obj.get("parameters") or {}
    units = {var: (meta.get(var) or {}).get("units") for var in values}
    coords = (obj.get("geometry") or {}).get("coordinates") or []
    if len(coords) >= 2:
        frame.attrs["lon"], frame.attrs["lat"] = float(coords[0]), float(coords[1])
    if len(coords) >= 3 and coords[2] is not None:
        frame.attrs["elevation"] = float(coords[2])
    return frame, units


def cell_path(location_id: str, cell_id: str):
    """Cache path data/raw/power/<loc>/<cell_id>.json."""
    return PATHS["raw_power"] / location_id / f"{cell_id}.json"


def load_cell(location_id: str, cell_id: str) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Parse one cached cell."""
    path = cell_path(location_id, cell_id)
    if not path.exists():
        raise FileNotFoundError(f"not downloaded yet: {rel_to_root(path)}")
    with open(path, encoding="utf-8") as fh:
        return parse_power_json(json.load(fh))


# ------------------------------------------------------------- download

def download_cell(
    location_id: str,
    spec: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    force: bool = False,
    session: Any = None,
) -> str:
    """Download one cell unless cached. Returns "downloaded" or "cached"."""
    out = cell_path(location_id, spec["cell_id"])
    if out.exists() and not force:
        return "cached"

    parameters = list(spec["parameters"])
    batches = [parameters[i:i + MAX_PARAMS_PER_REQUEST] for i in range(0, len(parameters), MAX_PARAMS_PER_REQUEST)]
    parts, urls = [], []
    for batch in batches:
        params = build_request_params(spec["lat"], spec["lon"], batch, cfg["start"], cfg["end"])
        urls.append(request_url(params))
        parts.append(fetch_json(params, session=session))
    data = merge_responses(parts)
    _check_response(data, parameters)

    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".part")
    partial.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    partial.replace(out)

    record_params: dict[str, Any] = {
        "location": location_id,
        "cell_id": spec["cell_id"],
        "lat": spec["lat"],
        "lon": spec["lon"],
        "parameters": parameters,
        "start": _ymd(cfg["start"]),
        "end": _ymd(cfg["end"]),
        "community": COMMUNITY,
        "requests": len(urls),
    }
    if len(urls) > 1:
        record_params["request_urls"] = urls
    try:
        append_record(out, "download", source_url=urls[0] if len(urls) == 1 else BASE_URL,
                      params=record_params)
    except Exception:
        out.unlink(missing_ok=True)
        raise
    log.info("downloaded %s", rel_to_root(out))
    return "downloaded"


def check_identical_neighbours(location_id: str, cfg: Mapping[str, Any] | None = None) -> list[str]:
    """Neighbour cells whose target series equals the site's (same solar grid cell)."""
    cfg = cfg if cfg is not None else load_config()
    target = cfg["target"]
    if not cell_path(location_id, SITE_CELL).exists():
        return []
    site, _ = load_cell(location_id, SITE_CELL)
    if target not in site:
        return []
    same: list[str] = []
    for cell_id in NEIGHBOR_CELLS:
        if not cell_path(location_id, cell_id).exists():
            continue
        neighbour, _ = load_cell(location_id, cell_id)
        if target not in neighbour:
            continue
        a, b = site[target].align(neighbour[target], join="inner")
        mask = a.notna() & b.notna()
        if mask.any() and float((a[mask] - b[mask]).abs().max()) < 1e-9:
            same.append(cell_id)
    if same:
        log.warning("%s: neighbour cells %s are identical to the site (same solar grid cell); "
                    "they will be dropped when aligning", location_id, same)
    return same


def download_location(
    location_id: str,
    cfg: Mapping[str, Any] | None = None,
    *,
    force: bool = False,
    session: Any = None,
) -> dict[str, Any]:
    """Download (or reuse) all 5 cells for one location."""
    cfg = cfg if cfg is not None else load_config()
    location = get_location(cfg, location_id)
    session = session if session is not None else _make_session()
    status = {
        spec["cell_id"]: download_cell(location_id, spec, cfg, force=force, session=session)
        for spec in cell_specs(location, cfg)
    }
    return {"location": location_id, "cells": status,
            "identical_neighbours": check_identical_neighbours(location_id, cfg)}


def download_all(cfg: Mapping[str, Any] | None = None, *, force: bool = False, session: Any = None) -> list[dict[str, Any]]:
    """Download every configured location; failures are reported, not raised."""
    cfg = cfg if cfg is not None else load_config()
    session = session if session is not None else _make_session()
    summaries = []
    for loc in cfg["locations"]:
        try:
            summaries.append(download_location(loc["id"], cfg, force=force, session=session))
        except PowerAPIError as exc:
            log.error("%s: %s", loc["id"], exc)
            summaries.append({"location": loc["id"], "error": str(exc)})
    return summaries


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Download NASA POWER daily data for the configured locations.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="download every location in config/settings.yaml")
    group.add_argument("--location", metavar="ID", help="download one location, e.g. tunis")
    parser.add_argument("--force", action="store_true", help="download again even if cached")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.location:
        try:
            get_location(cfg, args.location)
        except KeyError:
            known = ", ".join(loc["id"] for loc in cfg["locations"])
            print(f"unknown location: {args.location}. Choose from: {known}")
            return 1
        ids = [args.location]
    else:
        ids = [loc["id"] for loc in cfg["locations"]]

    session = _make_session()
    failures = 0
    for location_id in ids:
        try:
            summary = download_location(location_id, cfg, force=args.force, session=session)
        except PowerAPIError as exc:
            failures += 1
            print(f"{location_id:<14} FAILED: {exc}")
            continue
        cells = summary["cells"]
        new = sum(state == "downloaded" for state in cells.values())
        cached = sum(state == "cached" for state in cells.values())
        same = ", ".join(summary["identical_neighbours"]) or "none"
        print(f"{location_id:<14} downloaded {new}, cached {cached}, identical neighbours: {same}")
    print(f"done: {len(ids) - failures} ok, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
