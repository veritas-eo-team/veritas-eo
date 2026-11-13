"""Score any daily CSV with VERITAS (Phase 2: any-file mode).

    python -m src.score_file my_data.csv --lat 36.8 --lon 10.18 [--units kwh]

The CSV needs a date column and a daily value column (MJ/m^2/day by default, as
NASA POWER). Neighbours and context come from NASA POWER for the given coordinates
(cached under data/external/power, never written to the provenance log), or from
nb_* / ctx_* columns already in the file (offline). The first ref_days days (365
by default) are used to learn the site's normal behaviour and should be clean.
Output: results/score_<file>.csv with the C5 columns plus a plain-language
`reasons` column.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.common import PATHS, get_logger, load_config
from src.data import power
from src.data.align import ALIGN_DEFAULTS, NB_COLUMNS, S2_COLUMNS, context_vars, select_neighbours, validate_c1
from src.detect.context_model import ContextModel
from src.pipeline import ALARM_ACTIONS, run_pipeline

log = get_logger(__name__)

DATE_NAMES = ("date", "day", "time", "timestamp", "datetime")
VALUE_NAMES = ("target", "value", "radiation", "ghi", "allsky", "allsky_sfc_sw_dwn", "irradiance", "solar")
UNIT_FACTORS = {"mj": 1.0, "kwh": 3.6, "wm2": 0.0864}  # -> MJ/m^2/day (NASA POWER AG units)
UNIT_LABELS = {"mj": "MJ/m^2/day", "kwh": "kWh/m^2/day", "wm2": "W/m^2"}
MIN_VALID_DAYS = 60
LABELS = (("step_score", "sudden step"), ("drift_score", "slow drift"),
          ("replay_score", "replay or stuck values"), ("consistency_score", "physics check"))


class InputError(ValueError):
    """The input file or arguments cannot be scored; the message says why."""


# ------------------------------------------------------------------ reading

def read_input(path: str | Path, date_col: str | None = None, value_col: str | None = None,
               units: str = "mj") -> tuple[pd.Series, pd.DataFrame]:
    """CSV -> (daily target Series in MJ/m^2/day on a continuous index, extra nb_*/ctx_* columns)."""
    path = Path(path)
    if not path.exists():
        raise InputError(f"file not found: {path}")
    if units not in UNIT_FACTORS:
        raise InputError(f"units must be one of {sorted(UNIT_FACTORS)}")
    try:
        raw = pd.read_csv(path, sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001 - report any parse problem plainly
        raise InputError(f"could not read {path.name} as CSV: {exc}") from None
    if raw.empty or raw.shape[1] < 2:
        raise InputError("the file needs at least a date column and a value column")

    lower = {str(c).strip().lower(): c for c in raw.columns}
    dc = date_col or next((lower[n] for n in DATE_NAMES if n in lower), raw.columns[0])
    if dc not in raw.columns:
        raise InputError(f"date column {dc!r} not found; columns are {list(raw.columns)}")
    extra_cols = [c for c in raw.columns if str(c).startswith(("nb_", "ctx_"))]
    if value_col:
        vc = value_col
    else:
        vc = next((lower[n] for n in VALUE_NAMES if n in lower), None)
        if vc is None:
            numeric = [c for c in raw.columns if c != dc and c not in extra_cols
                       and pd.to_numeric(raw[c], errors="coerce").notna().mean() > 0.5]
            if len(numeric) != 1:
                raise InputError("cannot tell which column holds the daily values; "
                                 f"use --value-col (columns are {list(raw.columns)})")
            vc = numeric[0]
    if vc not in raw.columns:
        raise InputError(f"value column {vc!r} not found; columns are {list(raw.columns)}")

    dates = pd.to_datetime(raw[dc], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    if dates.isna().mean() > 0.05:
        raise InputError(f"column {dc!r} does not look like dates")
    keep = dates.notna().to_numpy()
    dates = dates[keep].dt.normalize()
    if dates.duplicated().any():
        raise InputError("found several rows for the same day; give one value per day (daily total or mean)")

    order = np.argsort(dates.to_numpy())
    idx = pd.DatetimeIndex(dates.to_numpy()[order])
    full = pd.date_range(idx[0], idx[-1], freq="D", name="date")
    values = pd.to_numeric(raw.loc[keep, vc], errors="coerce").to_numpy(dtype=float)[order] * UNIT_FACTORS[units]
    target = pd.Series(values, index=idx).reindex(full)
    target[target < 0] = np.nan
    if int(target.notna().sum()) < MIN_VALID_DAYS:
        raise InputError(f"need at least {MIN_VALID_DAYS} days with values")

    median = float(target.median())  # already converted to MJ/m^2/day
    if units != "wm2" and median > 45:
        shown = median / UNIT_FACTORS[units]
        raise InputError(f"median value {shown:.1f} is too high for {UNIT_LABELS[units]}; "
                         "for daily mean W/m^2 use --units wm2")
    if units == "mj" and median < 6:
        target.attrs["unit_warning"] = (f"median value {median:.1f} is low for MJ/m^2/day; "
                                        "if your data are in kWh/m^2/day, use --units kwh")

    extras = pd.DataFrame(index=full)
    for c in extra_cols:
        col = pd.to_numeric(raw.loc[keep, c], errors="coerce").to_numpy(dtype=float)[order]
        extras[c] = pd.Series(col, index=idx).reindex(full)
    target.name = "target"
    return target, extras


# ------------------------------------------------------------------ context

def fetch_context(lat: float, lon: float, start: Any, end: Any, cfg: Mapping[str, Any],
                  cache_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, Any]]:
    """Site cell (target + context vars) and neighbour target series from NASA POWER."""
    run_cfg = {**cfg, "start": pd.Timestamp(start).strftime("%Y-%m-%d"),
               "end": pd.Timestamp(end).strftime("%Y-%m-%d")}
    session = power._make_session()
    site, units, cands = None, {}, {}
    for spec in power.cell_specs({"lat": float(lat), "lon": float(lon)}, run_cfg):
        cache = Path(cache_dir) / f"{spec['cell_id']}.json" if cache_dir else None
        if cache is not None and cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
        else:
            params = list(spec["parameters"])
            parts = []
            for i in range(0, len(params), power.MAX_PARAMS_PER_REQUEST):
                query = power.build_request_params(spec["lat"], spec["lon"],
                                                   params[i:i + power.MAX_PARAMS_PER_REQUEST],
                                                   run_cfg["start"], run_cfg["end"])
                parts.append(power.fetch_json(query, session=session))
            data = power.merge_responses(parts)
            power._check_response(data, params)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        frame, u = power.parse_power_json(data)
        if spec["cell_id"] == power.SITE_CELL:
            site, units = frame, u
        elif cfg["target"] in frame:
            cands[spec["cell_id"]] = frame[cfg["target"]]
    return site, cands, units


def build_frame(target: pd.Series, cfg: Mapping[str, Any], *, extras: pd.DataFrame | None = None,
                site: pd.DataFrame | None = None, candidates: Mapping[str, pd.Series] | None = None,
                units: Mapping[str, Any] | None = None, lat: float | None = None, lon: float | None = None,
                name: str = "external") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Contract C1 frame for an external series (Sentinel-2 columns are NaN)."""
    index = pd.DatetimeIndex(target.index, name="date")
    n = len(index)
    tv = cfg["target"]
    data: dict[str, np.ndarray] = {"target": target.to_numpy(dtype=float)}
    info: dict[str, Any] = {"name": name, "neighbours": [], "neighbours_dropped": {}}

    if extras is not None and any(str(c).startswith("nb_") for c in extras.columns):
        nbs = sorted(c for c in extras.columns if str(c).startswith("nb_"))[:len(NB_COLUMNS)]
        for i, c in enumerate(nbs):
            data[f"nb_{i}"] = extras[c].to_numpy(dtype=float)
        for var in context_vars(cfg):
            col = f"ctx_{var}"
            data[col] = extras[col].to_numpy(dtype=float) if col in extras else np.full(n, np.nan)
        info.update(mode="columns from file", neighbours=nbs)
    elif site is not None:
        site = site.reindex(index)
        cands = {k: v.reindex(index).astype("float64") for k, v in (candidates or {}).items()}
        ref = site[tv].astype("float64") if tv in site else pd.Series(np.nan, index=index)
        kept, dropped = select_neighbours(ref, cands, max_missing=ALIGN_DEFAULTS["max_neighbor_missing"],
                                          tol=ALIGN_DEFAULTS["identical_tol"])
        for i, series in enumerate(kept.values()):
            data[f"nb_{i}"] = series.to_numpy(dtype=float)
        for var in context_vars(cfg):
            data[f"ctx_{var}"] = site[var].to_numpy(dtype=float) if var in site else np.full(n, np.nan)
        info.update(mode="NASA POWER download", neighbours=list(kept), neighbours_dropped=dropped)
    else:
        raise InputError("give --lat and --lon so neighbours and context can be downloaded, "
                         "or include nb_* and ctx_* columns in the file")

    for col in S2_COLUMNS:
        data[col] = np.full(n, np.nan)
    df = pd.DataFrame(data, index=index).astype("float64")
    df.attrs = {"location": name,
                "lat": float(lat) if lat is not None else float("nan"),
                "lon": float(lon) if lon is not None else float("nan"),
                "climate": "unknown", "split": "external", "target_var": str(tv),
                "units": (units or {}).get(tv) or "MJ/m^2/day"}
    validate_c1(df)
    return df, info


# ------------------------------------------------------------------ scoring

def explain(result: pd.DataFrame) -> pd.Series:
    """Plain-language reasons for every flagged day (empty string otherwise)."""
    alarm = result["action"].isin(ALARM_ACTIONS).to_numpy()
    reasons = []
    for i, (_, row) in enumerate(result.iterrows()):
        parts = [f"{label} {row[col]:.1f}" for col, label in LABELS
                 if pd.notna(row[col]) and row[col] >= 1.0]
        if alarm[i] or parts:
            if pd.notna(row["gap"]):
                parts.append(f"reads {np.expm1(row['gap']):+.1%} vs expected")
            if pd.notna(row["est_start"]):
                parts.append(f"likely started {pd.Timestamp(row['est_start']):%Y-%m-%d}")
        reasons.append("; ".join(parts))
    return pd.Series(reasons, index=result.index)


def alarm_episodes(result: pd.DataFrame) -> list[dict[str, Any]]:
    """Consecutive verify/quarantine runs with dates, length, type and estimated start."""
    a = result["action"].isin(ALARM_ACTIONS).to_numpy()
    idx, eps, i = result.index, [], 0
    while i < len(a):
        if not a[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(a) and a[j + 1]:
            j += 1
        types = result["attack_type"].iloc[i:j + 1].astype(str)
        types = types[types != "none"]
        eps.append({"start": idx[i], "end": idx[j], "days": j - i + 1,
                    "type": types.mode().iloc[0] if len(types) else "none",
                    "est_start": result["est_start"].iloc[i]})
        i = j + 1
    return eps


def score_series(target: pd.Series, cfg: Mapping[str, Any] | None = None, model: ContextModel | None = None, *,
                 extras: pd.DataFrame | None = None, lat: float | None = None, lon: float | None = None,
                 name: str = "external", cache: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run VERITAS on one external daily series. Returns (C5 result + reasons, info)."""
    cfg = cfg if cfg is not None else load_config()
    has_cols = extras is not None and any(str(c).startswith("nb_") for c in extras.columns)
    site = cands = units = None
    if not has_cols:
        if lat is None or lon is None:
            raise InputError("give --lat and --lon so neighbours and context can be downloaded, "
                             "or include nb_* and ctx_* columns in the file")
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise InputError("lat must be in [-90, 90] and lon in [-180, 180]")
        start, end = target.index[0], target.index[-1]
        cache_dir = (Path("data") / "external" / "power" /
                     f"{lat:.3f}_{lon:.3f}_{start:%Y%m%d}_{end:%Y%m%d}") if cache else None
        site, cands, units = fetch_context(lat, lon, start, end, cfg, cache_dir)
    model = model if model is not None else ContextModel.load()
    df, info = build_frame(target, cfg, extras=extras if has_cols else None, site=site, candidates=cands,
                           units=units, lat=lat, lon=lon, name=name)
    result = run_pipeline(df, model, cfg)
    result["reasons"] = explain(result)
    ref_days = int((cfg.get("context_model") or {}).get("ref_days") or 365)
    valid = int(df["target"].notna().sum())
    info.update(days=len(df), valid_days=valid, ref_days=ref_days, short_reference=valid < ref_days + 30)
    info["frame"] = df
    return result, info


# ------------------------------------------------------------------ CLI

def print_summary(result: pd.DataFrame, info: Mapping[str, Any], out: Path) -> None:
    print(f"file: {info['name']} | {info['days']} days ({info['valid_days']} with values) | context: {info['mode']}")
    dropped = f" (dropped: {info['neighbours_dropped']})" if info["neighbours_dropped"] else ""
    print(f"neighbours used: {len(info['neighbours'])}{dropped}")
    first = result.index[0]
    ref_end = min(first + pd.Timedelta(days=info["ref_days"] - 1), result.index[-1])
    print(f"reference period (learning this site's normal behaviour; should be clean): "
          f"{first:%Y-%m-%d} .. {ref_end:%Y-%m-%d}")
    if info["short_reference"]:
        print("warning: less than about a year of data, so results may be unreliable")
    counts = result["action"].value_counts()
    print("days per action: " + ", ".join(f"{k} {int(counts.get(k, 0))}"
                                          for k in ("accept", "deweight", "verify", "quarantine")))
    eps = alarm_episodes(result)
    if not eps:
        print("no alarms: the series agrees with its neighbours, physics and context")
    else:
        print(f"alarm episodes: {len(eps)} (longest first)")
        for e in sorted(eps, key=lambda e: -e["days"])[:5]:
            est = "" if pd.isna(e["est_start"]) else f", likely started {pd.Timestamp(e['est_start']):%Y-%m-%d}"
            print(f"  {e['start']:%Y-%m-%d} .. {e['end']:%Y-%m-%d} ({e['days']} days), type {e['type']}{est}")
        flagged = result.loc[result["action"].isin(ALARM_ACTIONS), "reasons"]
        print(f"first alarm ({flagged.index[0]:%Y-%m-%d}): {flagged.iloc[0]}")
    print(f"saved {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score a daily CSV with VERITAS.")
    ap.add_argument("csv", help="CSV with a date column and a daily value column")
    ap.add_argument("--lat", type=float, help="latitude of the sensor")
    ap.add_argument("--lon", type=float, help="longitude of the sensor")
    ap.add_argument("--date-col", help="name of the date column (auto-detected by default)")
    ap.add_argument("--value-col", help="name of the value column (auto-detected by default)")
    ap.add_argument("--units", choices=sorted(UNIT_FACTORS), default="mj",
                    help="mj = MJ/m2/day (default, as NASA POWER), kwh = kWh/m2/day, wm2 = daily mean W/m2")
    ap.add_argument("--out", help="output CSV (default: results/score_<file>.csv)")
    ap.add_argument("--no-cache", action="store_true", help="download NASA POWER data again")
    a = ap.parse_args(argv)

    cfg = load_config()
    try:
        target, extras = read_input(a.csv, a.date_col, a.value_col, a.units)
        if target.attrs.get("unit_warning"):
            print(f"warning: {target.attrs['unit_warning']}")
        result, info = score_series(target, cfg, extras=extras, lat=a.lat, lon=a.lon,
                                    name=Path(a.csv).stem, cache=not a.no_cache)
    except (InputError, FileNotFoundError, power.PowerAPIError) as exc:
        print(f"error: {exc}")
        return 2
    out = Path(a.out) if a.out else PATHS["results"] / f"score_{Path(a.csv).stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, date_format="%Y-%m-%d", index_label="date")
    print_summary(result, info, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())