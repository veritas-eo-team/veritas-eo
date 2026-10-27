"""Shared helpers for the Streamlit app: data and model loading, scenario runs, charts.

Everything runs offline from data/clean/ and models/. If either is missing the app falls
back to synthetic data and a model fitted on synthetic frames, with a visible banner.
"""
from __future__ import annotations

import base64
import sys
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.common import PATHS, load_config  # noqa: E402

TITLE = "Weather Data Trust Monitor"
ATTACKS: tuple[str, ...] = ("none", "step", "drift", "replay", "spike", "stuck")
ALARM_ACTIONS: tuple[str, ...] = ("verify", "quarantine")
ACTION_COLORS: dict[str, str] = {
    "accept": "#2e7d32", "deweight": "#f9a825", "verify": "#ef6c00", "quarantine": "#c62828",
}
ACTION_LABELS: dict[str, str] = {
    "accept": "ACCEPT", "deweight": "DE-WEIGHT", "verify": "VERIFY", "quarantine": "QUARANTINE",
}
CLRSKY = "ctx_CLRSKY_SFC_SW_DWN"


# ------------------------------------------------------------------ setup and loading

def page_setup(subtitle: str) -> None:
    """Must be the first Streamlit call of every page."""
    st.set_page_config(page_title=f"{TITLE} | {subtitle}", layout="wide")


@st.cache_resource
def get_cfg() -> dict:
    return load_config()


@st.cache_resource
def get_model() -> tuple[Any, str | None]:
    """(trained context model, None) or (None, reason)."""
    try:
        from src.detect.context_model import ContextModel
        return ContextModel.load(), None
    except Exception as exc:  # missing or outdated model: the app falls back to demo mode
        return None, str(exc)


@st.cache_resource
def synthetic_model() -> Any:
    from src.detect.context_model import ContextModel
    from src.synthetic import make_synthetic_frame
    frames = [make_synthetic_frame(location=f"train{i}", seed=i) for i in range(3)]
    return ContextModel(cfg=get_cfg()).fit(frames)


@st.cache_data
def list_locs(split: str | None = None) -> list[str]:
    """Locations of a split whose clean frame exists."""
    try:
        from src.data.align import clean_path, list_locations
        return [loc for loc in list_locations(split, get_cfg()) if clean_path(loc).exists()]
    except Exception:
        return []


@st.cache_data
def load_frame(location: str) -> pd.DataFrame:
    from src.data.align import load
    return load(location, get_cfg())


def get_inputs(location: str) -> tuple[pd.DataFrame, Any, str, bool]:
    """(frame, model, label, demo_mode) for a location; synthetic data if real data is missing."""
    model, _ = get_model()
    if model is not None and location in list_locs():
        try:
            return load_frame(location), model, f"{location} (NASA POWER)", False
        except Exception:
            pass
    from src.synthetic import make_synthetic_frame
    frame = make_synthetic_frame(location="demo", seed=7, split="test")
    return frame, synthetic_model(), "synthetic demo location", True


def demo_banner(demo: bool) -> None:
    if demo:
        st.warning("Demo mode: real data or the trained model is missing, so this page uses synthetic data.")


# ------------------------------------------------------------------ scenarios

def attack_and_window(df: pd.DataFrame, cfg: Mapping[str, Any], attack: str, size: float | None,
                      duration: int | None, noise: float, start: str | None, seed: int) -> tuple[pd.DataFrame, dict]:
    """Inject the attack on the full series and cut [start - warm-up, end of test year]."""
    from src.attacks.inject import inject

    ty = int(cfg["test_year"])
    ev = dict(cfg.get("eval") or {})
    window = ev.get("attack_window") or ["02-01", "09-30"]
    warmup = int(ev.get("warmup_days") or 365)
    kw: dict[str, Any] = {"seed": int(seed), "window_start": f"{ty}-{window[0]}", "window_end": f"{ty}-{window[1]}"}
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


@st.cache_data(show_spinner="Running the pipeline...")
def run_scenario(location: str, attack: str, size: float | None, duration: int | None,
                 noise: float, start: str | None, seed: int) -> tuple[pd.DataFrame, dict, pd.DataFrame, str, bool]:
    """(C5 result, attack labels, input frame, location label, demo_mode)."""
    from src.pipeline import run_pipeline

    cfg = get_cfg()
    df, model, label, demo = get_inputs(location)
    run_df, labels = attack_and_window(df, cfg, attack, size, duration, noise, start, seed)
    result = run_pipeline(run_df, model, cfg)
    return result, labels, run_df, label, demo


def summarize_run(result: pd.DataFrame, labels: Mapping[str, Any]) -> dict[str, Any]:
    """True start, first alarm after it, delay, detected type, start estimate, lowest trust."""
    alarm = result["action"].isin(ALARM_ACTIONS)
    out: dict[str, Any] = {"true_start": labels.get("start"), "first_alarm": None, "delay": None,
                           "type": None, "est_start": None, "lowest_trust": float(result["trust"].min()),
                           "alarm_days": int(alarm.sum())}
    start = labels.get("start")
    if start is None:
        return out
    after = result.index >= start
    out["lowest_trust"] = float(result.loc[after, "trust"].min())
    hits = alarm[after]
    if hits.any():
        first = hits[hits].index[0]
        out.update(first_alarm=first, delay=int((first - start).days),
                   type=str(result.loc[first, "attack_type"]), est_start=result.loc[first, "est_start"])
    return out


def scenario_controls(prefix: str) -> dict[str, Any]:
    """Sidebar controls; returns the arguments of run_scenario."""
    cfg = get_cfg()
    test, train = list_locs("test"), list_locs("train")
    options = test + train or ["demo"]
    names = {**{loc: f"{loc} (unseen test location)" for loc in test}, **{loc: f"{loc} (train)" for loc in train}}
    sb = st.sidebar
    location = sb.selectbox("Location", options, format_func=lambda loc: names.get(loc, loc), key=f"{prefix}_loc")
    attack = sb.selectbox("Attack", ATTACKS, index=1, key=f"{prefix}_attack")
    size: float | None = None
    duration: int | None = None
    noise = 0.0
    if attack == "step":
        size = sb.slider("Step size (%)", -30, 30, 12, key=f"{prefix}_size") / 100.0
    elif attack == "drift":
        size = sb.slider("Final drift size (%)", 1, 30, 15, key=f"{prefix}_size") / 100.0
        duration = sb.slider("Ramp duration (days)", 7, 180, 60, key=f"{prefix}_dur")
    elif attack == "replay":
        duration = sb.slider("Replay length (days)", 3, 21, 7, key=f"{prefix}_dur")
        noise = sb.slider("Noise on copied values (%)", 0.0, 2.0, 0.0, 0.1, key=f"{prefix}_noise") / 100.0
    elif attack == "spike":
        size = sb.slider("Spike size (%)", 10, 100, 50, key=f"{prefix}_size") / 100.0
        duration = sb.slider("Spike duration (days)", 1, 7, 1, key=f"{prefix}_dur")
    elif attack == "stuck":
        duration = sb.slider("Stuck duration (days)", 3, 21, 7, key=f"{prefix}_dur")
    ty = int(cfg["test_year"])
    start = None
    if attack != "none" and not sb.checkbox("Random start (seeded)", value=False, key=f"{prefix}_rand"):
        picked = sb.date_input("Attack start", value=date(ty, 5, 1), min_value=date(ty, 2, 1),
                               max_value=date(ty, 9, 30), key=f"{prefix}_start")
        start = str(picked)
    seed = int(sb.number_input("Seed", min_value=0, max_value=999, value=1, key=f"{prefix}_seed"))
    return {"location": location, "attack": attack, "size": size, "duration": duration,
            "noise": noise, "start": start, "seed": seed}


# ------------------------------------------------------------------ charts and widgets

def plot_run(result: pd.DataFrame, run_df: pd.DataFrame, labels: Mapping[str, Any],
             upto: pd.Timestamp | None = None, xlim: tuple | None = None) -> plt.Figure:
    """Observed vs expected (with clear-sky), gap, detector scores, trust coloured by action."""
    r = result if upto is None else result.loc[:upto]
    if xlim is not None:
        r = r.loc[xlim[0]:xlim[1]]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3.0, 1.4, 1.6, 1.2]})
    ax = axes[0]
    ax.plot(r.index, r["target"], lw=1.0, color="#1f4e79", label="observed")
    ax.plot(r.index, r["expected"], lw=1.0, color="#9e9e9e", label="expected from context")
    if CLRSKY in run_df:
        ax.plot(r.index, run_df[CLRSKY].reindex(r.index), lw=0.8, ls=":", color="#b8860b", label="clear-sky limit")
    ax.set_ylabel("kWh/mÂ²/day")
    ax.legend(loc="upper left", fontsize=8)

    gap_pct = (np.exp(r["gap"]) - 1.0) * 100.0
    axes[1].plot(r.index, gap_pct, lw=0.9, color="#37474f")
    axes[1].axhline(0, color="#9e9e9e", lw=0.6)
    axes[1].set_ylabel("gap (%)")

    for col, color in (("step_score", "#1565c0"), ("drift_score", "#6a1b9a"),
                       ("replay_score", "#00838f"), ("consistency_score", "#ef6c00")):
        axes[2].plot(r.index, r[col].clip(upper=2.5), lw=0.9, color=color, label=col.replace("_score", ""))
    axes[2].axhline(1.0, color="#c62828", ls="--", lw=0.8)
    axes[2].set_ylim(0, 2.6)
    axes[2].set_ylabel("scores")
    axes[2].legend(loc="upper left", fontsize=7, ncol=4)

    colors = [ACTION_COLORS.get(a, "#9e9e9e") for a in r["action"]]
    axes[3].bar(r.index, r["trust"], width=1.0, color=colors)
    for level in (0.8, 0.5, 0.2):
        axes[3].axhline(level, color="#bdbdbd", lw=0.5)
    axes[3].set_ylim(0, 1.02)
    axes[3].set_ylabel("trust")

    start, end = labels.get("start"), labels.get("end")
    if start is not None and len(r):
        right = min(pd.Timestamp(end), r.index[-1]) if end is not None else r.index[-1]
        if right >= start:
            for a in axes:
                a.axvspan(start, right, color="#c62828", alpha=0.07)
    if xlim is not None:
        axes[3].set_xlim(xlim[0], xlim[1])
    fig.tight_layout()
    return fig


def badge_html(action: str, day: pd.Timestamp, trust: float) -> str:
    color = ACTION_COLORS.get(action, "#616161")
    text = ACTION_LABELS.get(action, str(action).upper())
    return (f"<div style='background:{color};color:white;padding:16px;border-radius:10px;"
            f"text-align:center;font-size:32px;font-weight:700'>{text}"
            f"<div style='font-size:15px;font-weight:400'>{pd.Timestamp(day).date()} Â· trust {trust:.2f}</div></div>")


def record_checks(records: Sequence[Mapping[str, Any]], keys: Mapping[str, Any]) -> list[bool]:
    """Per record: recomputed hash, link to the previous record and signature all valid."""
    from src.trust.provenance import GENESIS_HASH, compute_record_hash

    out, prev = [], GENESIS_HASH
    for rec in records:
        ok = compute_record_hash(rec) == rec.get("record_hash") and rec.get("prev_hash") == prev
        key = keys.get(rec.get("public_key_id"))
        try:
            key.verify(base64.b64decode(rec["signature"]), bytes.fromhex(rec["record_hash"]))
        except Exception:
            ok = False
        out.append(bool(ok))
        prev = rec.get("record_hash")
    return out


def results_dir() -> Path:
    return PATHS["results"]
