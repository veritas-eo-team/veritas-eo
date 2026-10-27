"""End-to-end pipeline (contract C5).

Order: context model -> per-site calibration -> step, drift, replay, consistency
detectors -> fusion -> response policy. Everything is causal: the output for day t
uses only data up to day t (est_start may be refined after an alarm).

Start dates per alarm episode: each detector's attrs give the start of its FIRST
alarm. For later alarm episodes the pipeline re-estimates the start causally
(step: refine_start; drift: hinge fit; consistency: first violation in the window;
replay: copied-window or stuck-run start), so a false alarm early in the run does not
leave a stale start for a later attack. C5 est_start is filled from the first day with
action verify/quarantine onward, from the detector matching attack_type.

CLI: python -m src.pipeline --location tunis --attack step --size 0.12 --seed 1
"""
from __future__ import annotations

import argparse
import time
import tracemalloc
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.common import PATHS, get_logger, load_config
from src.detect import consistency, drift, replay, step
from src.detect.context_model import (
    DEFAULT_MIN_REF_DAYS,
    DEFAULT_REF_DAYS,
    ContextModel,
    calibrate_site,
)
from src.detect.fusion import fuse
from src.respond.policy import decide

log = get_logger(__name__)

C5_COLUMNS: list[str] = [
    "target", "expected", "sigma", "gap", "z",
    "step_score", "drift_score", "replay_score", "consistency_score",
    "p_compromise", "trust", "attack_type", "est_start", "action", "alert",
]
ALARM_ACTIONS: tuple[str, ...] = ("verify", "quarantine")


# ------------------------------------------------------------------ helpers

def _ref_params(cfg: Mapping[str, Any]) -> tuple[int, int]:
    section = dict(cfg.get("context_model") or {})
    ref = int(section.get("ref_days") or DEFAULT_REF_DAYS)
    min_ref = int(section.get("min_ref_days") or DEFAULT_MIN_REF_DAYS)
    return ref, min_ref


def _rising_edges(alarm: np.ndarray) -> np.ndarray:
    a = np.asarray(alarm, dtype=bool)
    prev = np.concatenate([[False], a[:-1]])
    return np.flatnonzero(a & ~prev)


def _per_episode(index: pd.DatetimeIndex, alarm: np.ndarray, first: Any,
                 later: Callable[[int], pd.Timestamp]) -> pd.Series:
    """Start date for each alarm episode, held from the episode's first day onward."""
    values: list[Any] = [pd.NaT] * len(index)
    current: Any = pd.NaT
    edges = set(_rising_edges(alarm).tolist())
    episode = 0
    for i in range(len(index)):
        if i in edges:
            current = first if (episode == 0 and first is not None) else later(i)
            episode += 1
        values[i] = current if alarm[i] else pd.NaT  # only while this detector alarms
    return pd.Series(pd.DatetimeIndex(values), index=index)


def _hinge_start(gap: np.ndarray, index: pd.DatetimeIndex, a: int, lookback: int) -> pd.Timestamp:
    lo = max(0, a - lookback + 1)
    seg = gap[lo:a + 1]
    pos = np.arange(lo, a + 1, dtype=float)
    ok = ~np.isnan(seg)
    fit = drift.fit_ramp(pos[ok], seg[ok], durations=(drift.HINGE_DAYS,)) if ok.sum() else None
    return index[int(fit["best_tau"])] if fit is not None else index[lo]


def _first_violation(violation: np.ndarray, index: pd.DatetimeIndex, a: int, window: int) -> pd.Timestamp:
    lo = max(0, a - window + 1)
    hits = np.flatnonzero(violation[lo:a + 1])
    return index[lo + int(hits[0])] if hits.size else index[a]


def _replay_starts(r_out: pd.DataFrame, target: pd.Series, rp: Mapping[str, Any]) -> pd.Series:
    """On replay alarm days: start of the copied window (or of the stuck run)."""
    index = target.index
    _, run_start = replay.stuck_runs(target.to_numpy(dtype=float), rp["decimals"], rp["stuck_run"])
    stuck = r_out["stuck"].to_numpy()
    values: list[Any] = [pd.NaT] * len(index)
    for i in np.flatnonzero(r_out["alarm"].to_numpy()):
        values[i] = index[run_start[i]] if stuck[i] else index[max(0, i - rp["m"] + 1)]
    return pd.Series(pd.DatetimeIndex(values), index=index)


def _c5_est_start(index: pd.DatetimeIndex, action: np.ndarray, attack_type: np.ndarray,
                  candidates: Mapping[str, pd.Series]) -> pd.Series:
    """est_start from the first verify/quarantine day onward, from the matching detector."""
    order = {
        "step": ("step", "drift", "consistency"),
        "drift": ("drift", "step", "consistency"),
        "replay": ("replay",),
        "stuck": ("replay",),
        "none": (),
    }
    values: list[Any] = [pd.NaT] * len(index)
    alarm_days = np.flatnonzero(np.isin(action, ALARM_ACTIONS))
    if alarm_days.size == 0:
        return pd.Series(pd.DatetimeIndex(values), index=index)
    prev: Any = pd.NaT
    is_alarm = np.isin(action, ALARM_ACTIONS)
    for i in range(int(alarm_days[0]), len(index)):
        if is_alarm[i] and (i == 0 or not is_alarm[i - 1]):
            prev = pd.NaT  # a new alarm episode starts fresh
        value = None
        for name in order.get(str(attack_type[i]), ()):
            c = candidates[name].iloc[i]
            if not pd.isna(c) and c <= index[i]:
                value = c
                break
        if value is None:
            value = prev if not pd.isna(prev) else index[i]
        values[i] = value
        prev = value
    return pd.Series(pd.DatetimeIndex(values), index=index)


# ------------------------------------------------------------------ pipeline

def run_pipeline(df: pd.DataFrame, model: ContextModel | None = None,
                 cfg: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Contract C5: one row per day of df with exactly C5_COLUMNS.

    If model is None, loads models/context_model.joblib (FileNotFoundError telling you to
    run: python -m src.detect.context_model --train). Works without ndvi/cloud or
    ctx_CLRSKY_SFC_SW_DWN.
    """
    cfg = cfg if cfg is not None else load_config()
    if "target" not in df:
        raise ValueError("df needs a target column (contract C1)")
    if model is None:
        model = ContextModel.load()
    index = df.index

    ref_days, min_ref = _ref_params(cfg)
    cal = calibrate_site(model.predict(df), ref_days, min_ref)
    z, gap = cal["z"], cal["gap"]
    gap_values = gap.to_numpy(dtype=float)

    sp = step.params_from_config(cfg)
    dp = drift.params_from_config(cfg)
    rp = replay.params_from_config(cfg)
    cp = consistency.params_from_config(cfg)
    s_out = step.detect(z, gap, sp)
    d_out = drift.detect(z, gap, dp)
    r_out = replay.detect(df["target"], z, rp)
    c_out = consistency.detect(df, gap, cp)
    violation = (c_out["over_clearsky"] | c_out["cloud_conflict"]).to_numpy()

    step_est = _per_episode(index, s_out["alarm"].to_numpy(), s_out.attrs.get("est_start"),
                            lambda e: step.refine_start(gap, index[e])[0] or index[e])
    drift_est = _per_episode(index, d_out["alarm"].to_numpy(), d_out.attrs.get("est_start"),
                             lambda e: _hinge_start(gap_values, index, e, int(dp["lookback"])))
    cons_est = _per_episode(index, c_out["alarm"].to_numpy(), c_out.attrs.get("est_start"),
                            lambda e: _first_violation(violation, index, e, cp["window"]))
    replay_est = _replay_starts(r_out, df["target"], rp)

    det = pd.DataFrame({
        "step_score": s_out["score"], "drift_score": d_out["score"],
        "replay_score": r_out["score"], "consistency_score": c_out["score"],
        "replay_exact": r_out["exact"], "stuck": r_out["stuck"], "gap": gap,
        "step_est_start": step_est, "drift_est_start": drift_est,
    }, index=index)
    fused = fuse(det, cfg)
    decided = decide(fused, cfg)

    est_start = _c5_est_start(
        index, decided["action"].to_numpy(), fused["attack_type"].to_numpy(),
        {"step": step_est, "drift": drift_est, "consistency": cons_est, "replay": replay_est},
    )
    out = pd.DataFrame({
        "target": df["target"].astype(float),
        "expected": cal["expected"], "sigma": cal["sigma"], "gap": gap, "z": z,
        "step_score": s_out["score"], "drift_score": d_out["score"],
        "replay_score": r_out["score"], "consistency_score": c_out["score"],
        "p_compromise": fused["p_compromise"], "trust": decided["trust"],
        "attack_type": fused["attack_type"], "est_start": est_start,
        "action": decided["action"], "alert": decided["alert"].astype(bool),
    }, index=index)
    return out[C5_COLUMNS]


def timed_run(df: pd.DataFrame, model: ContextModel | None = None,
              cfg: Mapping[str, Any] | None = None) -> tuple[pd.DataFrame, float, float]:
    """(result, ms per day, peak traced memory in MB). Model loading is not timed."""
    cfg = cfg if cfg is not None else load_config()
    model = model if model is not None else ContextModel.load()
    started_here = not tracemalloc.is_tracing()
    if started_here:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()
    t0 = time.perf_counter()
    try:
        result = run_pipeline(df, model, cfg)
    finally:
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        if started_here:
            tracemalloc.stop()
    return result, elapsed * 1000.0 / max(len(df), 1), peak / 2 ** 20


# ------------------------------------------------------------------ CLI

def _episodes(mask: pd.Series) -> int:
    return int(len(_rising_edges(mask.to_numpy())))


def _counts(actions: pd.Series) -> str:
    counts = actions.value_counts()
    return ", ".join(f"{a} {int(counts.get(a, 0))}" for a in ("accept", "deweight", "verify", "quarantine"))


def _day(ts: Any) -> str:
    return "-" if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d")


def main(argv: Sequence[str] | None = None) -> int:
    from src.attacks.inject import ATTACKS, attack_mask, inject
    from src.data.align import load

    parser = argparse.ArgumentParser(description="Run the trust pipeline on one location with one attack.")
    parser.add_argument("--location", required=True)
    parser.add_argument("--attack", default="step", choices=ATTACKS)
    parser.add_argument("--size", type=float, default=None)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--start", default=None, help="attack start YYYY-MM-DD (default: random in the eval window)")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = load_config()
    try:
        model = ContextModel.load()
    except FileNotFoundError as exc:
        print(exc)
        return 1
    df = load(args.location, cfg)
    ty = int(cfg["test_year"])
    ev = dict(cfg.get("eval") or {})
    window = ev.get("attack_window") or ["02-01", "09-30"]
    warmup = int(ev.get("warmup_days") or 365)

    attacked, labels = inject(df["target"], args.attack, size=args.size, start=args.start,
                              duration=args.duration, noise=args.noise, seed=args.seed,
                              window_start=f"{ty}-{window[0]}", window_end=f"{ty}-{window[1]}")
    anchor = labels["start"] if labels["start"] is not None else pd.Timestamp(f"{ty}-01-01")
    begin = max(df.index[0], anchor - pd.Timedelta(days=warmup))
    run_df = df.loc[begin:f"{ty}-12-31"].copy()
    run_df["target"] = attacked.loc[run_df.index]

    result, ms_per_day, peak_mb = timed_run(run_df, model, cfg)
    mask = attack_mask(labels, result.index)
    alarm = result["action"].isin(ALARM_ACTIONS)

    print(f"location {args.location} | attack {args.attack} size {labels['size']} | seed {args.seed}")
    print(f"run window {_day(result.index[0])} .. {_day(result.index[-1])} ({len(result)} days)")
    if labels["start"] is None:
        print(f"clean run: {_episodes(alarm)} alarm episodes, {int(alarm.sum())} alarm days")
        print(f"actions: {_counts(result['action'])}")
    else:
        start = labels["start"]
        before, after = result.index < start, result.index >= start
        print(f"true start {_day(start)}, last attacked day {_day(labels['end'])}")
        hits = alarm[after & alarm.to_numpy()]
        if len(hits):
            first = hits.index[0]
            est = result.loc[first, "est_start"]
            err = "-" if pd.isna(est) else f"{(est - start).days:+d} days"
            print(f"first alarm after start {_day(first)} (delay {(first - start).days} days), "
                  f"type {result.loc[first, 'attack_type']}, est_start {_day(est)} (error {err})")
        else:
            print("no alarm after the start (missed)")
        print(f"false-alarm episodes before start: {_episodes(alarm[before])}")
        fired = []
        for col in ("step_score", "drift_score", "replay_score", "consistency_score"):
            above = result.loc[after, col] >= 1.0
            if above.any():
                fired.append(f"{col.replace('_score', '')} ({_day(above[above].index[0])})")
        print(f"detectors that fired after start: {', '.join(fired) or 'none'}")
        print(f"actions before start: {_counts(result.loc[before, 'action'])}")
        print(f"actions after start:  {_counts(result.loc[after, 'action'])}")
        print(f"lowest trust after start: {result.loc[after, 'trust'].min():.3f}")
    print(f"cost: {ms_per_day:.3f} ms per day, peak memory {peak_mb:.1f} MB")

    out = result.copy()
    out["attacked"] = mask.to_numpy()
    path = PATHS["results"] / f"run_{args.location}_{args.attack}.csv"
    out.to_csv(path, date_format="%Y-%m-%d")
    print(f"saved results/{path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())