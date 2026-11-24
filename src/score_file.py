"""Score any daily CSV with VERITAS (Phase 2: any-file mode).

    python -m src.score_file my_data.csv --lat 36.8 --lon 10.18 [--units kwh] [--history auto] [--agg auto]

Input: a CSV with a date (or timestamp) column and a value column, in MJ/m^2/day by
default (NASA POWER units). Sub-daily files are aggregated to daily values (mean for
W/m^2, sum for energy per interval); days with under 80% of the usual number of
samples are set to missing.

Context: neighbours and site variables come from NASA POWER for the given
coordinates (cached under data/external/power, never written to the provenance log),
or from nb_* / ctx_* columns already in the file (offline).

No-oracle rule: the site's own target variable is NEVER downloaded for the dates being
scored. The site cell is requested with the context variables only, and a neighbour in
the site's own grid cell is detected by comparing it with the uploaded series.

Reference period (--history):
  nasa   the site's NASA POWER target for the ref_days BEFORE the file starts is the
         reference, so an attack anywhere in the file is measured against an earlier,
         independent baseline (default when coordinates are given). Assumes the file
         measures the same quantity as NASA POWER; a different instrument shows up as an
         offset from the first day and is reported.
  self   the first ref_days of the file are the reference (must be clean).
  short  under about a year of data and no NASA history: a third of the file
         (60-365 days) is the reference (must be clean).
Without coordinates or nb_* columns only the model-free replay and stuck-value checks
run (allow_no_context=True; the CLI always allows it).

Output: results/score_<file>.csv with the C5 columns plus a plain-language `reasons` column.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.common import PATHS, ROOT, get_logger, load_config
from src.data import power
from src.data.align import ALIGN_DEFAULTS, NB_COLUMNS, S2_COLUMNS, context_vars, validate_c1
from src.detect import replay
from src.detect.context_model import DEFAULT_MIN_REF_DAYS, DEFAULT_REF_DAYS, ContextModel
from src.detect.fusion import fuse
from src.pipeline import ALARM_ACTIONS, C5_COLUMNS, _c5_est_start, _replay_starts, run_pipeline
from src.respond.policy import decide

log = get_logger(__name__)

DATE_NAMES = ("date", "day", "time", "timestamp", "datetime")
VALUE_NAMES = ("target", "value", "radiation", "ghi", "allsky", "allsky_sfc_sw_dwn", "irradiance", "solar")
UNIT_FACTORS = {"mj": 1.0, "kwh": 3.6, "wm2": 0.0864}  # -> MJ/m^2/day (NASA POWER AG units)
UNIT_LABELS = {"mj": "MJ/m^2/day", "kwh": "kWh/m^2/day", "wm2": "W/m^2"}
MIN_VALID_DAYS = 60
LABELS = (("step_score", "sudden step"), ("drift_score", "slow drift"),
          ("replay_score", "replay or stuck values"), ("consistency_score", "physics check"))

AGG_MODES = ("auto", "mean", "sum", "none")
COVERAGE_MIN = 0.8          # share of the usual samples per day needed to aggregate a day
HISTORY_MODES = ("auto", "nasa", "self", "short")
MIN_HISTORY_DAYS = 300      # valid NASA days needed before the file (enough for the seasonal fit)
SELF_MARGIN_DAYS = 60       # "self" needs ref_days + this many valid days, else "short"
SAME_CELL_TOL = 0.02        # MJ/m^2/day; NASA values have 2 decimals, unit conversion adds rounding
SAME_CELL_SHARE = 0.5       # share of overlapping days within SAME_CELL_TOL that marks the same cell
MIN_OVERLAP = 30            # overlapping days needed for the same-cell test
START_OFFSET_WARN = 0.10    # |median gap| (log units, ~10%) at the file start that is reported
START_CHECK_DAYS = 60


class InputError(ValueError):
    """The input file or arguments cannot be scored; the message says why."""


class OracleGuardError(RuntimeError):
    """The code was about to use the site's own target for the dates being scored."""


# ------------------------------------------------------------------ reading

def read_input(path: str | Path, date_col: str | None = None, value_col: str | None = None,
               units: str = "mj", agg: str = "auto") -> tuple[pd.Series, pd.DataFrame]:
    """CSV -> (daily target Series in MJ/m^2/day on a continuous index, extra nb_*/ctx_* columns).

    Sub-daily rows are aggregated to days: agg "auto" averages W/m^2 and sums anything
    else (energy per interval); "none" rejects sub-daily files.
    """
    path = Path(path)
    if not path.exists():
        raise InputError(f"file not found: {path}")
    if units not in UNIT_FACTORS:
        raise InputError(f"units must be one of {sorted(UNIT_FACTORS)}")
    if agg not in AGG_MODES:
        raise InputError(f"agg must be one of {list(AGG_MODES)}")
    if agg == "sum" and units == "wm2":
        raise InputError("W/m^2 is a rate, so sub-daily values are averaged, not summed; use --agg mean")
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

    stamps = pd.to_datetime(raw[dc], errors="coerce")
    if getattr(stamps.dt, "tz", None) is not None:
        stamps = stamps.dt.tz_convert(None)
    if stamps.isna().mean() > 0.05:
        raise InputError(f"column {dc!r} does not look like dates")
    keep = stamps.notna().to_numpy()
    stamps = stamps[keep]
    if stamps.duplicated().any():
        raise InputError("found several rows for the same day and time; give one row per time step")
    values = pd.to_numeric(raw.loc[keep, vc], errors="coerce").to_numpy(dtype=float)
    days = stamps.dt.normalize()

    aggregation = None
    order = None
    if days.duplicated().any():
        if agg == "none":
            raise InputError("found several rows for the same day; give one value per day, "
                             "or use --agg mean / --agg sum")
        if extra_cols:
            raise InputError("nb_* and ctx_* columns are only supported in daily files")
        how = agg if agg != "auto" else ("mean" if units == "wm2" else "sum")
        frame = pd.DataFrame({"day": days.to_numpy(), "v": np.clip(values, 0.0, None)})
        per_day = frame.groupby("day")["v"]
        usual = int(frame.groupby("day").size().mode().iloc[0])
        daily = per_day.mean() if how == "mean" else per_day.sum(min_count=1)
        daily[per_day.count() < int(np.ceil(COVERAGE_MIN * usual))] = np.nan
        idx = pd.DatetimeIndex(daily.index)
        day_values = daily.to_numpy(dtype=float)
        aggregation = (f"{usual} samples per day -> daily {how}; days with under "
                       f"{COVERAGE_MIN:.0%} of them set to missing")
    else:
        order = np.argsort(days.to_numpy())
        idx = pd.DatetimeIndex(days.to_numpy()[order])
        day_values = values[order]

    full = pd.date_range(idx[0], idx[-1], freq="D", name="date")
    target = pd.Series(day_values * UNIT_FACTORS[units], index=idx).reindex(full)
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
    if order is not None:
        for c in extra_cols:
            col = pd.to_numeric(raw.loc[keep, c], errors="coerce").to_numpy(dtype=float)[order]
            extras[c] = pd.Series(col, index=idx).reindex(full)
    target.name = "target"
    if aggregation:
        target.attrs["aggregation"] = aggregation
    return target, extras


# ------------------------------------------------------------------ context

def _cache_dir(enabled: bool, lat: float, lon: float, start: Any, end: Any, kind: str) -> Path | None:
    if not enabled:
        return None
    return (ROOT / "data" / "external" / "power" /
            f"{lat:.3f}_{lon:.3f}_{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}" / kind)


def _cache_file(cache_dir: Path | None, cell_id: str, params: list[str]) -> Path | None:
    if cache_dir is None:
        return None
    tag = hashlib.sha1(",".join(params).encode("utf-8")).hexdigest()[:8]
    return Path(cache_dir) / f"{cell_id}_{tag}.json"


def _fetch_cell(lat: float, lon: float, params: list[str], start: str, end: str,
                session: Any, cache: Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One NASA POWER cell for exactly `params` -> (DataFrame, units); cached as JSON."""
    if cache is not None and cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        parts = []
        for i in range(0, len(params), power.MAX_PARAMS_PER_REQUEST):
            query = power.build_request_params(lat, lon, params[i:i + power.MAX_PARAMS_PER_REQUEST], start, end)
            parts.append(power.fetch_json(query, session=session))
        data = power.merge_responses(parts)
        power._check_response(data, params)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    frame, units = power.parse_power_json(data)
    return frame, dict(units)


def fetch_context(lat: float, lon: float, start: Any, end: Any, cfg: Mapping[str, Any],
                  cache_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, Any]]:
    """Site context variables and neighbour target series from NASA POWER.

    The site cell is requested WITHOUT the target variable (no-oracle rule); the target
    is also dropped from the result in case an old cache still holds it.
    """
    tv = cfg["target"]
    run_cfg = {**cfg, "start": pd.Timestamp(start).strftime("%Y-%m-%d"),
               "end": pd.Timestamp(end).strftime("%Y-%m-%d")}
    session = power._make_session()
    site, units, cands = None, {}, {}
    for spec in power.cell_specs({"lat": float(lat), "lon": float(lon)}, run_cfg):
        is_site = spec["cell_id"] == power.SITE_CELL
        params = [p for p in spec["parameters"] if not (is_site and p == tv)]
        if not params:
            continue
        frame, u = _fetch_cell(spec["lat"], spec["lon"], params, run_cfg["start"], run_cfg["end"],
                               session, _cache_file(cache_dir, spec["cell_id"], params))
        if is_site:
            site, units = frame.drop(columns=[tv], errors="ignore"), u
        elif tv in frame:
            cands[spec["cell_id"]] = frame[tv]
    if site is not None and tv in site.columns:
        raise OracleGuardError("site context still contains the target variable")
    return site, cands, units


def fetch_history(lat: float, lon: float, start: Any, end: Any, cfg: Mapping[str, Any],
                  cache_dir: Path | None = None) -> pd.Series:
    """The site's NASA POWER target for [start, end]. Callers keep `end` before the scored dates."""
    tv = cfg["target"]
    s, e = pd.Timestamp(start).strftime("%Y-%m-%d"), pd.Timestamp(end).strftime("%Y-%m-%d")
    run_cfg = {**cfg, "start": s, "end": e}
    spec = next(sp for sp in power.cell_specs({"lat": float(lat), "lon": float(lon)}, run_cfg)
                if sp["cell_id"] == power.SITE_CELL)
    frame, _ = _fetch_cell(spec["lat"], spec["lon"], [tv], s, e, power._make_session(),
                           _cache_file(cache_dir, spec["cell_id"], [tv]))
    index = pd.date_range(s, e, freq="D", name="date")
    if tv not in frame:
        return pd.Series(np.nan, index=index, name="target")
    return frame[tv].astype("float64").reindex(index).rename("target")


def _check_history(history_index: pd.DatetimeIndex, file_start: pd.Timestamp) -> None:
    """Raise OracleGuardError if any history day falls on or after the first scored day."""
    if len(history_index) and pd.Timestamp(history_index.max()) >= pd.Timestamp(file_start):
        raise OracleGuardError(
            f"history reaches {pd.Timestamp(history_index.max()):%Y-%m-%d}, "
            f"but scoring starts {pd.Timestamp(file_start):%Y-%m-%d}")


def pick_neighbours(target: pd.Series, candidates: Mapping[str, pd.Series], *,
                    max_missing: float = ALIGN_DEFAULTS["max_neighbor_missing"],
                    ) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Keep neighbours that are a different grid cell and mostly complete.

    The same-cell test uses the uploaded series itself, never a downloaded site value:
    a neighbour matching it within SAME_CELL_TOL on at least SAME_CELL_SHARE of the
    overlapping days is the site's own cell. Attacked days only lower the match share,
    so an attack covering under half the file cannot hide a same-cell neighbour.
    """
    kept: dict[str, pd.Series] = {}
    dropped: dict[str, str] = {}
    for cell_id, series in candidates.items():
        s = series.reindex(target.index).astype("float64")
        both = target.notna() & s.notna()
        if int(both.sum()) >= MIN_OVERLAP:
            share = float(((target[both] - s[both]).abs() <= SAME_CELL_TOL).mean())
            if share >= SAME_CELL_SHARE:
                dropped[cell_id] = f"same grid cell as the site (matches the file on {share:.0%} of days)"
                continue
        missing = float(s.isna().mean())
        if missing > max_missing:
            dropped[cell_id] = f"{missing:.0%} missing"
            continue
        kept[cell_id] = s
    return kept, dropped


def build_frame(target: pd.Series, cfg: Mapping[str, Any], *, extras: pd.DataFrame | None = None,
                site: pd.DataFrame | None = None, candidates: Mapping[str, pd.Series] | None = None,
                units: Mapping[str, Any] | None = None, lat: float | None = None, lon: float | None = None,
                name: str = "external") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Contract C1 frame for an external series (Sentinel-2 columns are NaN)."""
    index = pd.DatetimeIndex(target.index, name="date")
    target = pd.Series(target.to_numpy(dtype=float), index=index)
    n = len(index)
    tv = cfg["target"]
    data: dict[str, np.ndarray] = {"target": target.to_numpy(dtype=float)}
    info: dict[str, Any] = {"name": name, "neighbours": [], "neighbours_dropped": {}}

    if extras is not None and any(str(c).startswith("nb_") for c in extras.columns):
        nbs = sorted(c for c in extras.columns if str(c).startswith("nb_"))
        kept, dropped = pick_neighbours(target, {c: extras[c].reindex(index) for c in nbs})
        names = list(kept)[:len(NB_COLUMNS)]
        for i, c in enumerate(names):
            data[f"nb_{i}"] = kept[c].to_numpy(dtype=float)
        for var in context_vars(cfg):
            col = f"ctx_{var}"
            data[col] = extras[col].reindex(index).to_numpy(dtype=float) if col in extras else np.full(n, np.nan)
        info.update(mode="columns from file", neighbours=names, neighbours_dropped=dropped)
    elif site is not None:
        site = site.reindex(index)
        cands = {k: v.reindex(index).astype("float64") for k, v in (candidates or {}).items()}
        kept, dropped = pick_neighbours(target, cands)
        names = list(kept)[:len(NB_COLUMNS)]
        for i, c in enumerate(names):
            data[f"nb_{i}"] = kept[c].to_numpy(dtype=float)
        for var in context_vars(cfg):
            data[f"ctx_{var}"] = site[var].to_numpy(dtype=float) if var in site else np.full(n, np.nan)
        info.update(mode="NASA POWER download", neighbours=names, neighbours_dropped=dropped)
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


def resolve_history(history: str, *, has_coords: bool, has_cols: bool, valid_days: int, ref_days: int) -> str:
    """Pick the reference mode ("nasa", "self" or "short") for a run."""
    if history not in HISTORY_MODES:
        raise InputError(f"history must be one of {list(HISTORY_MODES)}")
    if history == "nasa" and (not has_coords or has_cols):
        raise InputError("--history nasa needs --lat/--lon and NASA POWER context (no nb_* columns in the file)")
    if history != "auto":
        return history
    if has_coords and not has_cols:
        return "nasa"
    return "self" if valid_days >= ref_days + SELF_MARGIN_DAYS else "short"


def short_ref_days(valid_days: int, ref_days: int = DEFAULT_REF_DAYS) -> int:
    """Reference length for short files: a third of the file, between 60 days and ref_days."""
    return int(max(2 * DEFAULT_MIN_REF_DAYS, min(ref_days, valid_days // 3)))


def score_without_context(target: pd.Series, cfg: Mapping[str, Any], *,
                          name: str = "external") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Model-free run (no coordinates, no nb_* columns): replay and stuck-value checks only."""
    index = pd.DatetimeIndex(target.index, name="date")
    t = pd.Series(target.to_numpy(dtype=float), index=index, name="target")
    rp = replay.params_from_config(cfg)
    r_out = replay.detect(t, None, rp)
    nan = pd.Series(np.nan, index=index)
    zero = pd.Series(0.0, index=index)
    nat = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    det = pd.DataFrame({
        "step_score": zero, "drift_score": zero, "replay_score": r_out["score"],
        "consistency_score": zero, "replay_exact": r_out["exact"], "stuck": r_out["stuck"],
        "gap": nan, "step_est_start": nat, "drift_est_start": nat,
    }, index=index)
    fused = fuse(det, cfg)
    decided = decide(fused.assign(gap=nan.to_numpy()), cfg)
    est = _c5_est_start(index, decided["action"].to_numpy(), fused["attack_type"].to_numpy(),
                        {"step": nat, "drift": nat, "consistency": nat,
                         "replay": _replay_starts(r_out, t, rp)})
    result = pd.DataFrame({
        "target": t, "expected": nan, "sigma": nan, "gap": nan, "z": nan,
        "step_score": zero, "drift_score": zero, "replay_score": r_out["score"],
        "consistency_score": zero, "p_compromise": fused["p_compromise"], "trust": decided["trust"],
        "attack_type": fused["attack_type"], "est_start": est,
        "action": decided["action"], "alert": decided["alert"].astype(bool),
    }, index=index)[C5_COLUMNS]
    result["reasons"] = explain(result)
    info = {
        "name": name, "mode": "no context (replay and stuck checks only)",
        "neighbours": [], "neighbours_dropped": {},
        "days": len(t), "valid_days": int(t.notna().sum()), "ref_days": 0, "history": "none",
        "reference": None, "reference_in_file": False, "short_reference": False,
        "notes": ["no coordinates and no nb_* columns: only the replay and stuck-value checks ran; "
                  "the step, drift and physics checks need --lat/--lon"],
        "aggregation": target.attrs.get("aggregation"), "frame": None,
    }
    return result, info


def score_series(target: pd.Series, cfg: Mapping[str, Any] | None = None, model: ContextModel | None = None, *,
                 extras: pd.DataFrame | None = None, lat: float | None = None, lon: float | None = None,
                 name: str = "external", cache: bool = True, history: str = "auto",
                 allow_no_context: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run VERITAS on one external daily series. Returns (C5 result + reasons, info).

    history: "auto" | "nasa" | "self" | "short" (see the module docstring). Without
    coordinates and nb_* columns this raises InputError, unless allow_no_context=True,
    in which case only the replay and stuck-value checks run.
    """
    cfg = cfg if cfg is not None else load_config()
    has_cols = extras is not None and any(str(c).startswith("nb_") for c in extras.columns)
    has_coords = lat is not None and lon is not None
    if not has_cols and not has_coords:
        if not allow_no_context:
            raise InputError("give --lat and --lon so neighbours and context can be downloaded, "
                             "or include nb_* and ctx_* columns in the file")
        return score_without_context(target, cfg, name=name)
    if has_coords and not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise InputError("lat must be in [-90, 90] and lon in [-180, 180]")

    ref_days = int((cfg.get("context_model") or {}).get("ref_days") or DEFAULT_REF_DAYS)
    valid = int(target.notna().sum())
    mode = resolve_history(history, has_coords=has_coords, has_cols=has_cols,
                           valid_days=valid, ref_days=ref_days)
    file_start, file_end = pd.Timestamp(target.index[0]), pd.Timestamp(target.index[-1])
    notes: list[str] = []
    full_target = pd.Series(target.to_numpy(dtype=float), index=pd.DatetimeIndex(target.index, name="date"),
                            name="target")

    if mode == "nasa":
        h_start = file_start - pd.Timedelta(days=ref_days)
        h_end = file_start - pd.Timedelta(days=1)
        _check_history(pd.date_range(h_start, h_end, freq="D"), file_start)
        try:
            fetched = fetch_history(lat, lon, h_start, h_end, cfg,
                                    _cache_dir(cache, lat, lon, h_start, h_end, "history"))
            _check_history(pd.DatetimeIndex(fetched.index), file_start)
            hist = fetched.astype("float64").reindex(pd.date_range(h_start, h_end, freq="D", name="date"))
            n_hist = int(hist.notna().sum())
            if n_hist < MIN_HISTORY_DAYS:
                raise InputError(f"only {n_hist} days of NASA POWER history before the file")
        except OracleGuardError:
            raise
        except Exception as exc:  # noqa: BLE001 - any download problem falls back to file-only calibration
            fallback = "self" if valid >= ref_days + SELF_MARGIN_DAYS else "short"
            notes.append(f"NASA POWER history not available ({exc}); used '{fallback}' calibration instead")
            log.warning("history download failed (%s); falling back to %s", exc, fallback)
            mode = fallback
        else:
            full_target = pd.concat([hist, full_target]).rename("target")
            full_target.index.name = "date"

    if mode == "nasa":
        run_ref = ref_days
        reference = (pd.Timestamp(full_target.index[0]), file_start - pd.Timedelta(days=1))
        in_file = False
    else:
        run_ref = ref_days if mode == "self" else short_ref_days(valid, ref_days)
        reference = (file_start, min(file_start + pd.Timedelta(days=run_ref - 1), file_end))
        in_file = True
        if mode == "short":
            notes.append(f"under about a year of data and no NASA history: the first {run_ref} days "
                         "are the reference and must be clean")

    site = cands = units = None
    if not has_cols:
        ctx_start = pd.Timestamp(full_target.index[0])
        site, cands, units = fetch_context(lat, lon, ctx_start, file_end, cfg,
                                           _cache_dir(cache, lat, lon, ctx_start, file_end, "context"))
    model = model if model is not None else ContextModel.load()
    df, info = build_frame(full_target, cfg, extras=extras if has_cols else None, site=site,
                           candidates=cands, units=units, lat=lat, lon=lon, name=name)

    run_cfg = {**cfg, "context_model": {**dict(cfg.get("context_model") or {}), "ref_days": int(run_ref)}}
    result = run_pipeline(df, model, run_cfg).loc[file_start:].copy()
    result["reasons"] = explain(result)

    if mode == "nasa":
        head = result["gap"].iloc[:START_CHECK_DAYS].to_numpy(dtype=float)
        head = head[~np.isnan(head)]
        if head.size >= 20 and abs(float(np.median(head))) > START_OFFSET_WARN:
            notes.append(f"the file reads {np.expm1(np.median(head)):+.0%} vs expected from its first days: "
                         "either a different instrument from NASA POWER (rerun with --history self) "
                         "or an attack already active when the file starts")

    info.update(days=len(target), valid_days=valid, ref_days=int(run_ref), history=mode,
                reference=reference, reference_in_file=in_file,
                short_reference=(mode == "short") or (in_file and valid < run_ref + 30),
                notes=notes, aggregation=target.attrs.get("aggregation"))
    info["frame"] = df.loc[file_start:]
    return result, info


# ------------------------------------------------------------------ CLI

def print_summary(result: pd.DataFrame, info: Mapping[str, Any], out: Path) -> None:
    print(f"file: {info['name']} | {info['days']} days ({info['valid_days']} with values) | context: {info['mode']}")
    if info.get("aggregation"):
        print(f"aggregated: {info['aggregation']}")
    if info["neighbours"] or info["neighbours_dropped"]:
        dropped = f" (dropped: {info['neighbours_dropped']})" if info["neighbours_dropped"] else ""
        print(f"neighbours used: {len(info['neighbours'])}{dropped}")
    ref = info.get("reference")
    if ref is not None:
        where = ("inside the file, should be clean" if info["reference_in_file"]
                 else "NASA POWER history before the file, never the scored days")
        print(f"reference period ({info['history']}): {ref[0]:%Y-%m-%d} .. {ref[1]:%Y-%m-%d} ({where})")
    for note in info.get("notes", []):
        print(f"note: {note}")
    if info.get("short_reference"):
        print("warning: short reference period, so results may be unreliable")
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
        main = max(eps, key=lambda e: e["days"])
        print(f"longest episode starts {main['start']:%Y-%m-%d}: {result.loc[main['start'], 'reasons']}")
    print(f"saved {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score a daily CSV with VERITAS.")
    ap.add_argument("csv", help="CSV with a date column and a value column (daily or sub-daily)")
    ap.add_argument("--lat", type=float, help="latitude of the sensor")
    ap.add_argument("--lon", type=float, help="longitude of the sensor")
    ap.add_argument("--date-col", help="name of the date column (auto-detected by default)")
    ap.add_argument("--value-col", help="name of the value column (auto-detected by default)")
    ap.add_argument("--units", choices=sorted(UNIT_FACTORS), default="mj",
                    help="mj = MJ/m2/day (default, as NASA POWER), kwh = kWh/m2/day, wm2 = mean W/m2")
    ap.add_argument("--agg", choices=AGG_MODES, default="auto",
                    help="sub-daily rows: auto = mean for wm2, sum otherwise; none = reject")
    ap.add_argument("--history", choices=HISTORY_MODES, default="auto",
                    help="reference period: nasa = NASA POWER year before the file (default with "
                         "coordinates), self = first year of the file, short = first third of the file")
    ap.add_argument("--out", help="output CSV (default: results/score_<file>.csv)")
    ap.add_argument("--no-cache", action="store_true", help="download NASA POWER data again")
    a = ap.parse_args(argv)

    cfg = load_config()
    try:
        target, extras = read_input(a.csv, a.date_col, a.value_col, a.units, a.agg)
        if target.attrs.get("unit_warning"):
            print(f"warning: {target.attrs['unit_warning']}")
        result, info = score_series(target, cfg, extras=extras, lat=a.lat, lon=a.lon,
                                    name=Path(a.csv).stem, cache=not a.no_cache,
                                    history=a.history, allow_no_context=True)
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