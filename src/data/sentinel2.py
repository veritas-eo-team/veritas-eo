"""Sentinel-2 L2A NDVI and cloud context from the Copernicus Data Space Statistical API.

For each location this asks the Statistical API for daily statistics over a small
box centred on the site and caches one row per acquisition date to
data/raw/s2/<loc>.csv with the columns:

    date, ndvi_mean (clear pixels only), cloud_fraction, valid_fraction

Credentials come from the environment variables CDSE_CLIENT_ID and
CDSE_CLIENT_SECRET. If they are missing, or the API fails, a warning is logged and
an empty frame with the right columns is returned: the rest of the system works
without satellite data.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

from src.common import PATHS, ROOT, get_logger, load_config
from src.trust.provenance import append_record

log = get_logger(__name__)

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
BASE_URL = "https://sh.dataspace.copernicus.eu"
STATS_URL = BASE_URL + "/api/v1/statistics"
CRS_WGS84 = "http://www.opengis.net/def/crs/EPSG/0/4326"

ENV_CLIENT_ID = "CDSE_CLIENT_ID"
ENV_CLIENT_SECRET = "CDSE_CLIENT_SECRET"

COLUMNS = ["ndvi_mean", "cloud_fraction", "valid_fraction"]
CLOUD_SCL_CLASSES = (3, 8, 9, 10)  # cloud shadow, cloud medium prob., cloud high prob., cirrus

# Thresholds Module 4 uses to keep an NDVI value (used here only for the CLI summary).
CLEAR_CLOUD_MAX = 0.3
CLEAR_VALID_MIN = 0.5

DEFAULTS: dict[str, Any] = {
    "bbox_km": 3.0,          # side of the square box around the site
    "resolution_m": 20.0,    # SCL is native 20 m; 20 m keeps processing-unit use low
    "collection": "sentinel-2-l2a",
    "retries": 3,
    "backoff_s": 2.0,
    "timeout_s": 120.0,
    "delay_s": 1.0,          # pause between requests
}

# Evalscript: one pass computes everything the statistics need.
#   dataMask   -> pixel has data and is not SCL 0 (no data) or 1 (saturated/defective)
#   cloud      -> 1 if SCL in {3, 8, 9, 10}; its mean over valid pixels = cloud fraction
#   clear      -> valid and not cloud
#   ndvi_clear -> NDVI on clear pixels, 0 elsewhere
# mean(ndvi_clear) / mean(clear) = mean NDVI over clear pixels only.
EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "SCL", "dataMask"] }],
    output: [
      { id: "ndvi_clear", bands: 1, sampleType: "FLOAT32" },
      { id: "clear", bands: 1, sampleType: "FLOAT32" },
      { id: "cloud", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function evaluatePixel(s) {
  var valid = (s.dataMask == 1 && s.SCL != 0 && s.SCL != 1) ? 1 : 0;
  var cloud = (s.SCL == 3 || s.SCL == 8 || s.SCL == 9 || s.SCL == 10) ? 1 : 0;
  var clear = valid * (1 - cloud);
  var denom = s.B08 + s.B04;
  var ndvi = denom > 0 ? (s.B08 - s.B04) / denom : 0;
  return {
    ndvi_clear: [ndvi * clear],
    clear: [clear],
    cloud: [cloud],
    dataMask: [valid]
  };
}
"""

# Indirection so tests can skip real waiting.
_sleep: Callable[[float], None] = time.sleep

# Possible PATHS keys for data/raw/s2 (Module 0 names); falls back to ROOT/data/raw/s2.
_PATH_KEYS = ("raw_s2", "s2", "s2_raw", "data_raw_s2")


class S2Error(RuntimeError):
    """A Copernicus request failed."""


class S2HTTPError(S2Error):
    """A non-retryable HTTP error from Copernicus."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


# ---------------------------------------------------------------------------
# Config, locations, paths
# ---------------------------------------------------------------------------

def s2_config(cfg: dict) -> dict:
    """Sentinel-2 settings: DEFAULTS overridden by the optional `sentinel2` config section."""
    merged = dict(DEFAULTS)
    merged.update(cfg.get("sentinel2") or {})
    return merged


def _locations(cfg: dict) -> dict[str, dict]:
    """Map location id -> location dict; accepts a list of dicts or a dict keyed by id."""
    locs = cfg.get("locations") or []
    out: dict[str, dict] = {}
    if isinstance(locs, dict):
        for key, value in locs.items():
            if isinstance(value, dict) and "lat" in value:
                out[str(value.get("id", key))] = {**value, "id": value.get("id", key)}
            elif isinstance(value, list):  # grouped by split
                for item in value:
                    out[str(item["id"])] = item
    else:
        for item in locs:
            out[str(item["id"])] = item
    return out


def location_info(cfg: dict, loc: str) -> dict:
    """Return the config entry for one location (KeyError if unknown)."""
    locs = _locations(cfg)
    if loc not in locs:
        raise KeyError(f"unknown location {loc!r}; known: {', '.join(locs)}")
    return locs[loc]


def _to_date(value: Any) -> dt.date:
    """Accept date, datetime, Timestamp or 'YYYY-MM-DD' string."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value).strip()[:10])


def s2_dir() -> Path:
    """Folder for cached Sentinel-2 CSVs (created on demand)."""
    for key in _PATH_KEYS:
        if key in PATHS:
            folder = Path(PATHS[key])
            break
    else:
        folder = Path(ROOT) / "data" / "raw" / "s2"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def csv_path(loc: str) -> Path:
    """data/raw/s2/<loc>.csv"""
    return s2_dir() / f"{loc}.csv"


def get_credentials() -> tuple[str, str] | None:
    """(client_id, client_secret) from the environment, or None if either is missing."""
    cid = os.environ.get(ENV_CLIENT_ID, "").strip()
    secret = os.environ.get(ENV_CLIENT_SECRET, "").strip()
    if not cid or not secret:
        return None
    return cid, secret


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def bbox_around(lat: float, lon: float, size_km: float) -> list[float]:
    """[min_lon, min_lat, max_lon, max_lat] of a size_km x size_km box centred on the site."""
    half = size_km / 2.0
    dlat = half / 110.574
    dlon = half / (111.320 * max(math.cos(math.radians(lat)), 1e-6))
    return [
        lon - dlon,
        max(-90.0, lat - dlat),
        lon + dlon,
        min(90.0, lat + dlat),
    ]


def year_ranges(start: Any, end: Any) -> list[tuple[str, str]]:
    """Split [start, end] (inclusive dates) into per-year [from, to) ISO UTC ranges."""
    d0, d1 = _to_date(start), _to_date(end)
    if d1 < d0:
        raise ValueError(f"end {d1} is before start {d0}")
    out = []
    for year in range(d0.year, d1.year + 1):
        a = max(d0, dt.date(year, 1, 1))
        b = min(d1, dt.date(year, 12, 31)) + dt.timedelta(days=1)
        out.append((f"{a.isoformat()}T00:00:00Z", f"{b.isoformat()}T00:00:00Z"))
    return out


def build_request_body(bbox: list[float], time_from: str, time_to: str, s2cfg: dict) -> dict:
    """Statistical API request: daily statistics of EVALSCRIPT over bbox."""
    pixels = max(1, int(round(float(s2cfg["bbox_km"]) * 1000.0 / float(s2cfg["resolution_m"]))))
    return {
        "input": {
            "bounds": {"bbox": [float(v) for v in bbox], "properties": {"crs": CRS_WGS84}},
            "data": [{"type": str(s2cfg["collection"])}],
        },
        "aggregation": {
            "timeRange": {"from": time_from, "to": time_to},
            "aggregationInterval": {"of": "P1D"},
            "evalscript": EVALSCRIPT,
            "width": pixels,
            "height": pixels,
        },
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _post(session: Any, url: str, **kwargs: Any) -> Any:
    poster = session.post if session is not None else requests.post
    return poster(url, **kwargs)


def _short(resp: Any) -> str:
    text = getattr(resp, "text", "") or ""
    return str(text)[:200]


def _post_json(
    session: Any,
    url: str,
    *,
    retries: int,
    backoff_s: float,
    timeout_s: float,
    **kwargs: Any,
) -> dict:
    """POST and return the JSON object. Retries network errors, bad JSON, 429 and 5xx."""
    last = "no attempt made"
    for attempt in range(retries + 1):
        try:
            resp = _post(session, url, timeout=timeout_s, **kwargs)
        except requests.RequestException as exc:
            last = f"network error ({type(exc).__name__})"
        else:
            status = int(resp.status_code)
            if status == 200:
                try:
                    obj = resp.json()
                except ValueError:
                    last = "response was not valid JSON"
                else:
                    if isinstance(obj, dict):
                        return obj
                    last = "response JSON is not an object"
            elif status == 429 or status >= 500:
                last = f"HTTP {status}"
            else:
                raise S2HTTPError(status, _short(resp))
        if attempt < retries:
            wait = backoff_s * (2 ** attempt)
            log.warning("request to %s failed (%s); retrying in %.0f s", url, last, wait)
            _sleep(wait)
    raise S2Error(f"request to {url} failed after {retries + 1} attempts: {last}")


class TokenProvider:
    """OAuth2 client-credentials token, cached until shortly before it expires."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        session: Any = None,
        token_url: str = TOKEN_URL,
        retries: int = 3,
        backoff_s: float = 2.0,
        timeout_s: float = 60.0,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._session = session
        self._token_url = token_url
        self._retries = retries
        self._backoff_s = backoff_s
        self._timeout_s = timeout_s
        self._token: str | None = None
        self._expires_at = 0.0

    def __repr__(self) -> str:
        return "TokenProvider(<credentials hidden>)"

    def get(self, force: bool = False) -> str:
        """Return a valid access token, requesting a new one if needed."""
        now = time.monotonic()
        if not force and self._token and now < self._expires_at - 30:
            return self._token
        obj = _post_json(
            self._session,
            self._token_url,
            retries=self._retries,
            backoff_s=self._backoff_s,
            timeout_s=self._timeout_s,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
        )
        token = obj.get("access_token")
        if not isinstance(token, str) or not token:
            raise S2Error("token response has no access_token")
        try:
            expires_in = float(obj.get("expires_in") or 300)
        except (TypeError, ValueError):
            expires_in = 300.0
        self._token = token
        self._expires_at = time.monotonic() + expires_in
        log.info("obtained Copernicus access token (valid %.0f s)", expires_in)
        return token


def fetch_stats(body: dict, tokens: TokenProvider, session: Any = None, s2cfg: dict | None = None) -> dict:
    """POST one Statistical API request; on HTTP 401 refresh the token once and retry."""
    s2cfg = s2cfg if s2cfg is not None else dict(DEFAULTS)
    kw = {
        "retries": int(s2cfg["retries"]),
        "backoff_s": float(s2cfg["backoff_s"]),
        "timeout_s": float(s2cfg["timeout_s"]),
    }
    for attempt in range(2):
        headers = {
            "Authorization": f"Bearer {tokens.get(force=attempt > 0)}",
            "Content-Type": "application/json",
        }
        try:
            return _post_json(session, STATS_URL, json=body, headers=headers, **kw)
        except S2HTTPError as exc:
            if exc.status == 401 and attempt == 0:
                log.info("access token rejected; requesting a new one")
                continue
            raise
    raise S2Error("statistics request failed after token refresh")


# ---------------------------------------------------------------------------
# Parsing and storage
# ---------------------------------------------------------------------------

def empty_frame() -> pd.DataFrame:
    """Empty frame with the Sentinel-2 columns and a DatetimeIndex named 'date'."""
    idx = pd.DatetimeIndex([], name="date")
    return pd.DataFrame(index=idx, columns=COLUMNS, dtype="float64")


def _num(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _band_stats(outputs: dict, name: str) -> dict:
    bands = outputs[name]["bands"]
    band = bands["B0"] if "B0" in bands else next(iter(bands.values()))
    return band["stats"]


def parse_stats(obj: dict) -> pd.DataFrame:
    """Statistical API JSON -> DataFrame[ndvi_mean, cloud_fraction, valid_fraction] by date.

    Intervals with an error or with no valid pixels are skipped.
    """
    if not isinstance(obj, dict) or not isinstance(obj.get("data"), list):
        raise ValueError("not a Statistical API response: missing 'data' list")
    rows: dict[pd.Timestamp, tuple[float, float, float]] = {}
    n_errors = 0
    for entry in obj["data"]:
        if entry.get("error"):
            n_errors += 1
            continue
        day = pd.Timestamp(str(entry["interval"]["from"])[:10])
        outputs = entry["outputs"]
        cloud = _band_stats(outputs, "cloud")
        sample = _num(cloud.get("sampleCount"))
        nodata = _num(cloud.get("noDataCount"))
        if math.isnan(sample) or sample <= 0:
            continue
        if math.isnan(nodata):
            nodata = 0.0
        valid_fraction = min(1.0, max(0.0, (sample - nodata) / sample))
        if valid_fraction <= 0.0:
            continue
        cloud_fraction = _num(cloud.get("mean"))
        clear_mean = _num(_band_stats(outputs, "clear").get("mean"))
        ndvi_clear_mean = _num(_band_stats(outputs, "ndvi_clear").get("mean"))
        if clear_mean > 0:
            ndvi = min(1.0, max(-1.0, ndvi_clear_mean / clear_mean))
        else:
            ndvi = math.nan
        rows[day] = (ndvi, cloud_fraction, valid_fraction)
    if n_errors:
        log.warning("%d interval(s) returned an error and were skipped", n_errors)
    if not rows:
        return empty_frame()
    idx = pd.DatetimeIndex(sorted(rows), name="date")
    return pd.DataFrame([rows[d] for d in idx], index=idx, columns=COLUMNS, dtype="float64")


def _combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if not p.empty]
    if not parts:
        return empty_frame()
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "date"
    return df.astype("float64")


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index).strftime("%Y-%m-%d")
    out.index.name = "date"
    tmp = path.with_name(path.name + ".tmp")
    out.to_csv(tmp, float_format="%.6f")
    os.replace(tmp, path)


def load_s2(loc: str) -> pd.DataFrame:
    """Read data/raw/s2/<loc>.csv; empty frame (right columns) if the file is missing."""
    path = csv_path(loc)
    if not path.exists():
        return empty_frame()
    raw = pd.read_csv(path)
    missing = [c for c in ["date", *COLUMNS] if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    idx = pd.DatetimeIndex(pd.to_datetime(raw["date"]), name="date")
    df = pd.DataFrame(raw[COLUMNS].to_numpy(dtype="float64"), index=idx, columns=COLUMNS)
    return df.sort_index()


def count_clear(df: pd.DataFrame, cloud_max: float = CLEAR_CLOUD_MAX, valid_min: float = CLEAR_VALID_MIN) -> int:
    """Number of acquisitions Module 4 would keep as clear NDVI."""
    if df.empty:
        return 0
    mask = (
        df["ndvi_mean"].notna()
        & (df["cloud_fraction"] <= cloud_max)
        & (df["valid_fraction"] >= valid_min)
    )
    return int(mask.sum())


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _make_tokens(creds: tuple[str, str], session: Any, s2cfg: dict) -> TokenProvider:
    return TokenProvider(
        creds[0],
        creds[1],
        session=session,
        retries=int(s2cfg["retries"]),
        backoff_s=float(s2cfg["backoff_s"]),
        timeout_s=float(s2cfg["timeout_s"]),
    )


def download_location(
    loc: str,
    cfg: dict | None = None,
    *,
    force: bool = False,
    session: Any = None,
    tokens: TokenProvider | None = None,
    log_path: Path | None = None,
    keys_dir: Path | None = None,
    root: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Download (or load cached) Sentinel-2 statistics for one location.

    Returns (frame, state) with state in {"downloaded", "cached", "no_credentials", "failed"}.
    On "no_credentials" or "failed" the frame is empty and nothing is cached.
    """
    cfg = cfg if cfg is not None else load_config()
    s2cfg = s2_config(cfg)
    info = location_info(cfg, loc)
    path = csv_path(loc)

    if path.exists() and not force:
        log.info("%s: using cached %s", loc, path.name)
        return load_s2(loc), "cached"

    if tokens is None:
        creds = get_credentials()
        if creds is None:
            log.warning(
                "%s: %s / %s not set; skipping Sentinel-2 (the system works without it)",
                loc, ENV_CLIENT_ID, ENV_CLIENT_SECRET,
            )
            return empty_frame(), "no_credentials"
        tokens = _make_tokens(creds, session, s2cfg)

    lat, lon = float(info["lat"]), float(info["lon"])
    bbox = bbox_around(lat, lon, float(s2cfg["bbox_km"]))
    start, end = _to_date(cfg["start"]), _to_date(cfg["end"])

    parts: list[pd.DataFrame] = []
    n_requests = 0
    try:
        for i, (t0, t1) in enumerate(year_ranges(start, end)):
            if i > 0 and float(s2cfg["delay_s"]) > 0:
                _sleep(float(s2cfg["delay_s"]))
            body = build_request_body(bbox, t0, t1, s2cfg)
            obj = fetch_stats(body, tokens, session, s2cfg)
            n_requests += 1
            part = parse_stats(obj)
            parts.append(part)
            log.info("%s: %s -> %d acquisitions", loc, t0[:4], len(part))
    except (S2Error, ValueError, KeyError, TypeError) as exc:
        log.warning("%s: Sentinel-2 download failed (%s); continuing without satellite data", loc, exc)
        return empty_frame(), "failed"

    df = _combine(parts)
    _write_csv_atomic(df, path)

    params = {
        "location": loc,
        "lat": lat,
        "lon": lon,
        "bbox": [round(float(v), 6) for v in bbox],
        "bbox_km": float(s2cfg["bbox_km"]),
        "resolution_m": float(s2cfg["resolution_m"]),
        "collection": str(s2cfg["collection"]),
        "aggregation": "P1D",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "evalscript_sha256": hashlib.sha256(EVALSCRIPT.encode("utf-8")).hexdigest(),
        "requests": n_requests,
        "rows": int(len(df)),
    }
    try:
        append_record(
            path, "download", STATS_URL, params,
            log_path=log_path, keys_dir=keys_dir, root=root,
        )
    except Exception as exc:  # never keep a file the log does not vouch for
        path.unlink(missing_ok=True)
        log.error("%s: provenance record could not be written (%s); removed %s", loc, exc, path.name)
        return empty_frame(), "failed"

    log.info("%s: saved %d acquisitions to %s", loc, len(df), path.name)
    return df, "downloaded"


def download_all(cfg: dict | None = None, *, force: bool = False, session: Any = None) -> list[dict]:
    """Download every configured location. Returns one summary dict per location."""
    cfg = cfg if cfg is not None else load_config()
    s2cfg = s2_config(cfg)
    own_session = session is None
    session = session if session is not None else requests.Session()
    creds = get_credentials()
    tokens = _make_tokens(creds, session, s2cfg) if creds else None
    results: list[dict] = []
    try:
        for loc in _locations(cfg):
            try:
                df, state = download_location(loc, cfg, force=force, session=session, tokens=tokens)
            except Exception as exc:  # keep going with the other locations
                log.error("%s: unexpected error (%s)", loc, exc)
                df, state = empty_frame(), "failed"
            results.append(
                {"location": loc, "state": state, "rows": int(len(df)), "clear_images": count_clear(df)}
            )
    finally:
        if own_session:
            session.close()
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Sentinel-2 NDVI and cloud statistics.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="all configured locations")
    group.add_argument("--location", help="one location id, e.g. tunis")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.all:
        results = download_all(cfg, force=args.force)
    else:
        try:
            df, state = download_location(args.location, cfg, force=args.force)
        except KeyError as exc:
            print(f"error: {exc}")
            return 1
        results = [{"location": args.location, "state": state, "rows": len(df), "clear_images": count_clear(df)}]

    print(f"{'location':<16}{'state':<16}{'rows':>6}{'clear':>7}")
    for r in results:
        print(f"{r['location']:<16}{r['state']:<16}{r['rows']:>6}{r['clear_images']:>7}")
    if any(r["state"] == "no_credentials" for r in results):
        print(f"\nSet {ENV_CLIENT_ID} and {ENV_CLIENT_SECRET} to download Sentinel-2 data.")
    return 1 if any(r["state"] == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())