"""Align, clean and load per-location frames (contract C1).

build_clean() reads the cached NASA POWER cells and the Sentinel-2 CSV, puts them
on one continuous daily index (config start..end) and writes
data/clean/<loc>.parquet, recorded in the provenance log as a "derived" record
whose params hold the SHA-256 of every input file.

Column rules:
  target            site value of the target variable (fill values -> NaN)
  nb_0..nb_3        target at the neighbour cells that were kept, renumbered in
                    N, E, S, W order; a neighbour is dropped if it is identical to
                    the site (same solar grid cell) or has > 30% missing days
  ctx_<VAR>         other NASA variables at the site
  ndvi              last clear Sentinel-2 NDVI, forward-filled only (causal)
  cloud             Sentinel-2 cloud fraction on acquisition days with enough
                    valid pixels, else NaN
  days_since_clear  days since the last clear NDVI (NaN before the first one)

Which raw cell each nb_i came from, the units and the input hashes are stored in
the parquet metadata (read_info()), so df.attrs stays exactly as in C1.

CLI: python -m src.data.align --all | --location tunis [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common import PATHS, get_location, get_logger, load_config, rel_to_root
from src.data import power
from src.data.sentinel2 import csv_path as s2_csv_path
from src.data.sentinel2 import load_s2
from src.trust.provenance import append_record, canonical_json, file_sha256

log = get_logger(__name__)

FORMAT_VERSION = 1
META_KEY = b"eo_trust"
NEIGHBOR_SLOTS = 4
NB_COLUMNS: tuple[str, ...] = tuple(f"nb_{i}" for i in range(NEIGHBOR_SLOTS))
S2_COLUMNS: tuple[str, ...] = ("ndvi", "cloud", "days_since_clear")
ATTR_KEYS: tuple[str, ...] = ("location", "lat", "lon", "climate", "split", "target_var", "units")
SPLITS: tuple[str, ...] = ("train", "test")

ALIGN_DEFAULTS: dict[str, float] = {
    "max_neighbor_missing": 0.30,  # drop a neighbour with more missing days than this
    "identical_tol": 1e-9,         # max abs difference that counts as "identical to site"
    "s2_cloud_max": 0.30,          # NDVI kept only if cloud_fraction <= this
    "s2_valid_min": 0.50,          # ... and valid_fraction >= this (also needed for `cloud`)
}


# ------------------------------------------------------------------ config

def align_config(cfg: Mapping[str, Any]) -> dict[str, float]:
    """ALIGN_DEFAULTS overridden by the optional `align` config section."""
    merged = dict(ALIGN_DEFAULTS)
    merged.update(cfg.get("align") or {})
    return {key: float(value) for key, value in merged.items()}


def list_locations(split: str | None = None, cfg: Mapping[str, Any] | None = None) -> list[str]:
    """Location ids in config order, optionally only one split ("train" or "test")."""
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS} or None")
    cfg = cfg if cfg is not None else load_config()
    return [str(loc["id"]) for loc in cfg["locations"] if split is None or loc["split"] == split]


def daily_index(cfg: Mapping[str, Any]) -> pd.DatetimeIndex:
    """Continuous daily index from config start to end (inclusive)."""
    return pd.date_range(str(cfg["start"]), str(cfg["end"]), freq="D", name="date")


def context_vars(cfg: Mapping[str, Any]) -> list[str]:
    """Context variables in config order, without the target."""
    return [var for var in cfg["context_vars"] if var != cfg["target"]]


def clean_path(location_id: str) -> Path:
    """data/clean/<loc>.parquet"""
    return PATHS["clean"] / f"{location_id}.parquet"


# ------------------------------------------------------------------ C1 check

def validate_c1(df: pd.DataFrame) -> None:
    """Raise ValueError if `df` does not match contract C1."""
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.name != "date":
        raise ValueError("index must be a DatetimeIndex named 'date'")
    if idx.tz is not None:
        raise ValueError("index must be tz-naive")
    if len(idx) > 1 and not bool((idx[1:] - idx[:-1] == pd.Timedelta(days=1)).all()):
        raise ValueError("index must be continuous and daily")
    cols = list(df.columns)
    for required in ("target", *S2_COLUMNS):
        if required not in cols:
            raise ValueError(f"missing column: {required}")
    for col in cols:
        if not (col == "target" or col in NB_COLUMNS or col.startswith("ctx_") or col in S2_COLUMNS):
            raise ValueError(f"unexpected column: {col}")
    nbs = [c for c in cols if c in NB_COLUMNS]
    if nbs != list(NB_COLUMNS[: len(nbs)]):
        raise ValueError("neighbour columns must be nb_0..nb_k without gaps")
    not_float = [c for c in cols if df[c].dtype != np.float64]
    if not_float:
        raise ValueError(f"columns must be float64: {not_float}")
    missing_attrs = [key for key in ATTR_KEYS if key not in df.attrs]
    if missing_attrs:
        raise ValueError(f"missing attrs: {missing_attrs}")


# ------------------------------------------------------------------ building blocks

def select_neighbours(
    site_target: pd.Series,
    candidates: Mapping[str, pd.Series],
    *,
    max_missing: float = ALIGN_DEFAULTS["max_neighbor_missing"],
    tol: float = ALIGN_DEFAULTS["identical_tol"],
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Keep neighbours that differ from the site and are mostly complete.

    Returns (kept {cell_id: series}, dropped {cell_id: reason}), both in input order.
    """
    kept: dict[str, pd.Series] = {}
    dropped: dict[str, str] = {}
    for cell_id, series in candidates.items():
        both = site_target.notna() & series.notna()
        if both.any() and float((site_target[both] - series[both]).abs().max()) < tol:
            dropped[cell_id] = "identical to site"
            continue
        missing = float(series.isna().mean())
        if missing > max_missing:
            dropped[cell_id] = f"{missing:.0%} missing"
            continue
        kept[cell_id] = series
    return kept, dropped


def s2_features(
    s2: pd.DataFrame | None,
    index: pd.DatetimeIndex,
    *,
    cloud_max: float = ALIGN_DEFAULTS["s2_cloud_max"],
    valid_min: float = ALIGN_DEFAULTS["s2_valid_min"],
) -> pd.DataFrame:
    """Sentinel-2 rows -> DataFrame[ndvi, cloud, days_since_clear] on `index`.

    Causal: NDVI is forward-filled only, so day t uses acquisitions up to day t.
    Acquisitions outside `index` are ignored.
    """
    out = pd.DataFrame(np.nan, index=index, columns=list(S2_COLUMNS))
    if s2 is None or s2.empty:
        return out
    s2 = s2.copy()
    s2.index = pd.DatetimeIndex(s2.index).normalize()
    s2 = s2[~s2.index.duplicated(keep="last")].reindex(index)

    valid = s2["valid_fraction"].to_numpy(dtype=float)
    cloud = s2["cloud_fraction"].to_numpy(dtype=float)
    ndvi = s2["ndvi_mean"].to_numpy(dtype=float)
    usable = valid >= valid_min
    clear = usable & ~np.isnan(ndvi) & (cloud <= cloud_max)

    out["cloud"] = np.where(usable, np.clip(cloud, 0.0, 1.0), np.nan)
    out["ndvi"] = pd.Series(np.where(clear, ndvi, np.nan), index=index).ffill().to_numpy()
    position = np.arange(len(index), dtype=float)
    last_clear = pd.Series(np.where(clear, position, np.nan), index=index).ffill().to_numpy()
    out["days_since_clear"] = position - last_clear
    return out


def _input_files(location_id: str) -> list[Path]:
    """Raw files this location's clean frame depends on (site cell must exist)."""
    site = power.cell_path(location_id, power.SITE_CELL)
    if not site.exists():
        raise FileNotFoundError(
            f"{location_id}: no NASA POWER data; run: python -m src.data.power --location {location_id}"
        )
    files = [site]
    for cell_id in power.NEIGHBOR_CELLS:
        path = power.cell_path(location_id, cell_id)
        if path.exists():
            files.append(path)
    s2 = s2_csv_path(location_id)
    if s2.exists():
        files.append(s2)
    return files


def _hash_inputs(paths: list[Path]) -> dict[str, str]:
    return {rel_to_root(path): file_sha256(path) for path in paths}


def _build_key(inputs: Mapping[str, str], cfg: Mapping[str, Any], acfg: Mapping[str, float]) -> str:
    """Fingerprint of inputs and settings; unchanged key -> no rebuild needed."""
    settings = {
        "start": str(cfg["start"]),
        "end": str(cfg["end"]),
        "target": cfg["target"],
        "context_vars": list(cfg["context_vars"]),
        "align": dict(acfg),
    }
    blob = canonical_json({"format": FORMAT_VERSION, "inputs": dict(inputs), "settings": settings})
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _assemble(location_id: str, cfg: Mapping[str, Any], acfg: Mapping[str, float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the C1 frame (without attrs) and its info dict."""
    index = daily_index(cfg)
    target_var = cfg["target"]

    site, units = power.load_cell(location_id, power.SITE_CELL)
    if target_var not in site:
        raise ValueError(f"{location_id}: site cell has no {target_var}")
    site = site.reindex(index)
    target = site[target_var].astype("float64")

    candidates: dict[str, pd.Series] = {}
    unavailable: dict[str, str] = {}
    for cell_id in power.NEIGHBOR_CELLS:
        if not power.cell_path(location_id, cell_id).exists():
            unavailable[cell_id] = "not downloaded"
            continue
        frame, _ = power.load_cell(location_id, cell_id)
        if target_var not in frame:
            unavailable[cell_id] = f"no {target_var}"
            continue
        candidates[cell_id] = frame[target_var].reindex(index).astype("float64")
    kept, rejected = select_neighbours(
        target, candidates,
        max_missing=acfg["max_neighbor_missing"], tol=acfg["identical_tol"],
    )
    reasons = {**unavailable, **rejected}
    dropped = {cell: reasons[cell] for cell in power.NEIGHBOR_CELLS if cell in reasons}

    data: dict[str, pd.Series] = {"target": target}
    mapping: dict[str, str] = {}
    for i, (cell_id, series) in enumerate(kept.items()):
        data[f"nb_{i}"] = series
        mapping[f"nb_{i}"] = cell_id
    for var in context_vars(cfg):
        if var in site:
            data[f"ctx_{var}"] = site[var]
        else:
            log.warning("%s: site cell has no %s; ctx_%s is all NaN", location_id, var, var)
            data[f"ctx_{var}"] = pd.Series(np.nan, index=index)

    s2 = load_s2(location_id)
    features = s2_features(s2, index, cloud_max=acfg["s2_cloud_max"], valid_min=acfg["s2_valid_min"])
    for col in S2_COLUMNS:
        data[col] = features[col]

    df = pd.DataFrame(data, index=index).astype("float64")
    in_range = 0
    if not s2.empty:
        days = pd.DatetimeIndex(s2.index).normalize()
        in_range = int(((days >= index[0]) & (days <= index[-1])).sum())
    info: dict[str, Any] = {
        "location": location_id,
        "target_var": target_var,
        "units": units.get(target_var),
        "neighbours": mapping,
        "neighbours_dropped": dropped,
        "target_missing": float(target.isna().mean()),
        "s2_acquisitions": in_range,
        "clear_images": int((df["days_since_clear"] == 0).sum()),
        "start": index[0].strftime("%Y-%m-%d"),
        "end": index[-1].strftime("%Y-%m-%d"),
        "rows": int(len(df)),
    }
    return df, info


def _with_attrs(df: pd.DataFrame, cfg: Mapping[str, Any], location_id: str, units: str | None) -> pd.DataFrame:
    loc = get_location(cfg, location_id)
    df.attrs = {
        "location": location_id,
        "lat": float(loc["lat"]),
        "lon": float(loc["lon"]),
        "climate": str(loc["climate"]),
        "split": str(loc["split"]),
        "target_var": str(cfg["target"]),
        "units": units,
    }
    return df


def _write_parquet(df: pd.DataFrame, info: Mapping[str, Any], path: Path) -> None:
    out = df.copy()
    out.attrs = {}
    table = pa.Table.from_pandas(out, preserve_index=True)
    meta = dict(table.schema.metadata or {})
    meta[META_KEY] = json.dumps(info, sort_keys=True).encode("utf-8")
    table = table.replace_schema_metadata(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    pq.write_table(table, partial)
    os.replace(partial, path)


def read_info(location_id: str) -> dict[str, Any] | None:
    """Build info stored in the parquet metadata (neighbour mapping, units, inputs), or None."""
    path = clean_path(location_id)
    if not path.exists():
        return None
    raw = (pq.read_schema(path).metadata or {}).get(META_KEY)
    return json.loads(raw) if raw else None


# ------------------------------------------------------------------ public API

def build_location(
    location_id: str,
    cfg: Mapping[str, Any] | None = None,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build (or reuse) the clean frame. Returns (frame, info with state "built"|"unchanged").

    Rebuilds only when an input file or a relevant setting changed, or with force=True.
    Each rebuild writes the parquet and appends one "derived" provenance record.
    """
    cfg = cfg if cfg is not None else load_config()
    get_location(cfg, location_id)  # KeyError for unknown ids
    acfg = align_config(cfg)
    inputs = _hash_inputs(_input_files(location_id))
    key = _build_key(inputs, cfg, acfg)
    path = clean_path(location_id)

    if not force:
        old = read_info(location_id)
        if old is not None and old.get("build_key") == key:
            return load(location_id, cfg), {**old, "state": "unchanged"}

    df, info = _assemble(location_id, cfg, acfg)
    info.update(format=FORMAT_VERSION, build_key=key, inputs=inputs)
    _write_parquet(df, info, path)
    params = {
        "location": location_id,
        "inputs": inputs,
        "build_key": key,
        "neighbours": info["neighbours"],
        "neighbours_dropped": info["neighbours_dropped"],
        "rows": info["rows"],
        "clear_images": info["clear_images"],
    }
    try:
        append_record(path, "derived", None, params)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    log.info("%s: wrote %s", location_id, rel_to_root(path))
    return _with_attrs(df, cfg, location_id, info["units"]), {**info, "state": "built"}


def build_clean(location_id: str, cfg: Mapping[str, Any] | None = None, *, force: bool = False) -> pd.DataFrame:
    """Build data/clean/<loc>.parquet and return the C1 frame."""
    return build_location(location_id, cfg, force=force)[0]


def load(location_id: str, cfg: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Contract C1: read data/clean/<loc>.parquet (building it if missing), attrs from config."""
    cfg = cfg if cfg is not None else load_config()
    path = clean_path(location_id)
    if not path.exists():
        return build_clean(location_id, cfg)
    table = pq.read_table(path)
    raw = (table.schema.metadata or {}).get(META_KEY)
    info = json.loads(raw) if raw else {}
    df = table.to_pandas()
    df.attrs = {}
    idx = pd.DatetimeIndex(df.index, name="date")
    if len(idx):
        expected = pd.date_range(idx[0], idx[-1], freq="D", name="date")
        if not idx.equals(expected):
            raise ValueError(f"{path.name}: index is not continuous daily; rebuild with --force")
        idx = expected
    df.index = idx
    df = df.astype("float64")
    df = _with_attrs(df, cfg, location_id, info.get("units"))
    validate_c1(df)
    return df


# ------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Build clean, aligned frames (data/clean/<loc>.parquet).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="every location in config/settings.yaml")
    group.add_argument("--location", metavar="ID", help="one location, e.g. tunis")
    parser.add_argument("--force", action="store_true", help="rebuild even if inputs are unchanged")
    args = parser.parse_args(argv)

    cfg = load_config()
    ids = list_locations(cfg=cfg) if args.all else [args.location]
    print(f"{'location':<14}{'split':<6}{'state':<10}{'miss%':>6}  {'neighbours kept':<26}{'S2 clear':>8}  date range")
    failures = 0
    for location_id in ids:
        try:
            split = get_location(cfg, location_id)["split"]
            _, info = build_location(location_id, cfg, force=args.force)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            failures += 1
            print(f"{location_id:<14}FAILED: {exc}")
            continue
        kept = info["neighbours"]
        kept_txt = f"{len(kept)}/4 " + (",".join(kept.values()) or "-")
        print(
            f"{location_id:<14}{split:<6}{info['state']:<10}{100 * info['target_missing']:>6.1f}  "
            f"{kept_txt:<26}{info['clear_images']:>8}  {info['start']}..{info['end']}"
        )
        for cell_id, reason in info["neighbours_dropped"].items():
            print(f"{'':<14}  dropped {cell_id}: {reason}")
    print(f"done: {len(ids) - failures} ok, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())