"""Shared helpers for the VERITAS Streamlit app: branding, data and model loading,
scenario runs, interactive charts and day-level explanations.

Everything runs offline from data/clean/ and models/. If either is missing the app falls
back to synthetic data and a model fitted on synthetic frames, with a visible banner.
"""
from __future__ import annotations

import base64
import io
import sys
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import altair as alt  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.common import PATHS, load_config  # noqa: E402

BRAND = "VERITAS"
TAGLINE = "Weather data trust monitor"
TITLE = f"{BRAND} · {TAGLINE}"
ATTACKS: tuple[str, ...] = ("none", "step", "drift", "replay", "spike", "stuck")
ALARM_ACTIONS: tuple[str, ...] = ("verify", "quarantine")
ACTION_COLORS: dict[str, str] = {
    "accept": "#22c55e", "deweight": "#eab308", "verify": "#f97316", "quarantine": "#ef4444",
}
ACTION_LABELS: dict[str, str] = {
    "accept": "ACCEPT", "deweight": "DE-WEIGHT", "verify": "VERIFY", "quarantine": "QUARANTINE",
}
DETECTORS: tuple[tuple[str, str, str, str], ...] = (
    ("step", "step_score", "Step (CUSUM)", "a sustained shift has built up in the gap"),
    ("drift", "drift_score", "Drift (EWMA + slope)", "the gap is trending slowly away from zero"),
    ("replay", "replay_score", "Replay / stuck", "these days repeat earlier data or a frozen value"),
    ("consistency", "consistency_score", "Physics checks", "repeated clear-sky or cloud contradictions"),
)
DETECTOR_COLORS = ["#60a5fa", "#c084fc", "#2dd4bf", "#fb923c"]
CLRSKY = "ctx_CLRSKY_SFC_SW_DWN"
CHART_WIDTH = 950

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1380px;}
.vx-hero {background: linear-gradient(120deg, #0f2a4a 0%, #123a63 45%, #0e7490 100%);
  border: 1px solid #1f3b5c; border-radius: 16px; padding: 22px 28px; margin-bottom: 18px;}
.vx-hero h1 {margin: 0; font-size: 2.4rem; letter-spacing: 0.2em; color: #e0f2fe;}
.vx-hero p {margin: 6px 0 0 0; color: #bae6fd; font-size: 1.02rem;}
.vx-card {background: #131c2e; border: 1px solid #22314d; border-radius: 14px; padding: 14px 16px;
  height: 100%; min-height: 104px;}
.vx-card .label {color: #94a3b8; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.08em;}
.vx-card .value {color: #f1f5f9; font-size: 1.5rem; font-weight: 700; margin-top: 2px;}
.vx-card .note {font-size: 0.82rem; margin-top: 3px; color: #94a3b8;}
.vx-ok {color: #4ade80 !important;} .vx-bad {color: #f87171 !important;} .vx-mid {color: #facc15 !important;}
.vx-reason {background: #131c2e; border-left: 4px solid #38bdf8; border-radius: 8px; padding: 9px 13px;
  margin: 6px 0; color: #e2e8f0;}
.vx-reason.strong {border-left-color: #ef4444;}
.vx-reason.medium {border-left-color: #f97316;}
.vx-brand {font-weight: 800; letter-spacing: 0.24em; font-size: 1.35rem; color: #7dd3fc;}
.vx-sub {color: #94a3b8; font-size: 0.85rem; margin-bottom: 8px;}
</style>
"""


# ------------------------------------------------------------------ branding and layout

def page_setup(subtitle: str) -> None:
    """Must be the first Streamlit call of every page: config, style, sidebar brand."""
    st.set_page_config(page_title=f"{BRAND} | {subtitle}", page_icon="🛰️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown(f"<div class='vx-brand'>{BRAND}</div><div class='vx-sub'>{TAGLINE}</div>",
                        unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='vx-hero'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def card(container: Any, label: str, value: str, note: str = "", tone: str = "") -> None:
    tone_class = f" vx-{tone}" if tone else ""
    container.markdown(f"<div class='vx-card'><div class='label'>{label}</div><div class='value'>{value}</div>"
                       f"<div class='note{tone_class}'>{note}</div></div>", unsafe_allow_html=True)


def demo_banner(demo: bool) -> None:
    if demo:
        st.warning("Demo mode: real data or the trained model is missing, so this page uses synthetic data.")


# ------------------------------------------------------------------ loading

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
    """(frame, model, label, demo_mode); synthetic data if real data is missing."""
    model, _ = get_model()
    if model is not None and location in list_locs():
        try:
            return load_frame(location), model, f"{location} (NASA POWER)", False
        except Exception:
            pass
    from src.synthetic import make_synthetic_frame
    frame = make_synthetic_frame(location="demo", seed=7, split="test")
    return frame, synthetic_model(), "synthetic demo location", True


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
    names = {**{loc: f"{loc} · unseen test location" for loc in test}, **{loc: f"{loc} · train" for loc in train}}
    sb = st.sidebar
    sb.markdown("#### Scenario")
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


# ------------------------------------------------------------------ interactive chart

def _chart_frame(result: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({
        "date": result.index,
        "observed": result["target"].to_numpy(),
        "expected": result["expected"].to_numpy(),
        "gap_pct": ((np.exp(result["gap"]) - 1.0) * 100.0).to_numpy(),
        "trust": result["trust"].to_numpy(),
        "action": result["action"].to_numpy(),
        "attack_type": result["attack_type"].to_numpy(),
    })
    for name, col, _, _ in DETECTORS:
        d[name] = result[col].clip(upper=2.5).to_numpy()
    d["clear_sky"] = run_df[CLRSKY].reindex(result.index).to_numpy() if CLRSKY in run_df else np.nan
    return d


def altair_run(result: pd.DataFrame, run_df: pd.DataFrame, labels: Mapping[str, Any],
               upto: pd.Timestamp | None = None, xlim: tuple | None = None) -> alt.VConcatChart:
    """Four linked panels: values, gap, detector scores, trust. Scroll to zoom, drag to pan, hover for values."""
    r = result if upto is None else result.loc[:upto]
    if xlim is not None:
        r = r.loc[xlim[0]:xlim[1]]
    d = _chart_frame(r, run_df)
    lo = pd.Timestamp(xlim[0]) if xlim is not None else d["date"].min()
    hi = pd.Timestamp(xlim[1]) if xlim is not None else d["date"].max()
    x = alt.X("date:T", title=None, scale=alt.Scale(domain=[lo.isoformat(), hi.isoformat()]))
    when = alt.Tooltip("date:T", title="date")
    zoom = alt.selection_interval(bind="scales", encodings=["x"])

    band = None
    start, end = labels.get("start"), labels.get("end")
    if start is not None and end is not None:
        s, e = max(pd.Timestamp(start), lo), min(pd.Timestamp(end), hi)
        if s <= e:
            band = alt.Chart(pd.DataFrame({"start": [s], "end": [e]})).mark_rect(
                color="#ef4444", opacity=0.13).encode(x="start:T", x2="end:T")

    def banded(chart: Any) -> Any:
        return alt.layer(band, chart) if band is not None else chart

    values = d.melt(id_vars="date", value_vars=["observed", "expected"], var_name="series", value_name="value")
    lines = alt.Chart(values).mark_line(strokeWidth=1.5).encode(
        x=x, y=alt.Y("value:Q", title="kWh/m²/day"),
        color=alt.Color("series:N", scale=alt.Scale(domain=["observed", "expected"], range=["#38bdf8", "#94a3b8"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[when, "series:N", alt.Tooltip("value:Q", format=".2f")])
    clear = alt.Chart(d).mark_line(strokeDash=[4, 3], strokeWidth=1, color="#facc15").encode(
        x=x, y="clear_sky:Q", tooltip=[when, alt.Tooltip("clear_sky:Q", title="clear-sky limit", format=".2f")])
    top = banded(alt.layer(lines, clear)).add_params(zoom).properties(width=CHART_WIDTH, height=230)

    gap_line = alt.Chart(d).mark_line(strokeWidth=1, color="#cbd5e1").encode(
        x=x, y=alt.Y("gap_pct:Q", title="gap (%)"), tooltip=[when, alt.Tooltip("gap_pct:Q", title="gap %", format="+.1f")])
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="#475569").encode(y="y:Q")
    gap = banded(alt.layer(zero, gap_line)).properties(width=CHART_WIDTH, height=95)

    scores_long = d.melt(id_vars="date", value_vars=[n for n, _, _, _ in DETECTORS], var_name="detector", value_name="score")
    score_lines = alt.Chart(scores_long).mark_line(strokeWidth=1.3).encode(
        x=x, y=alt.Y("score:Q", title="scores", scale=alt.Scale(domain=[0, 2.6])),
        color=alt.Color("detector:N", scale=alt.Scale(domain=[n for n, _, _, _ in DETECTORS], range=DETECTOR_COLORS),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[when, "detector:N", alt.Tooltip("score:Q", format=".2f")])
    alarm_line = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color="#ef4444", strokeDash=[5, 4]).encode(y="y:Q")
    scores = banded(alt.layer(score_lines, alarm_line)).properties(width=CHART_WIDTH, height=125)

    trust = banded(alt.Chart(d).mark_bar().encode(
        x=x, y=alt.Y("trust:Q", title="trust", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("action:N", scale=alt.Scale(domain=list(ACTION_COLORS), range=list(ACTION_COLORS.values())),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[when, alt.Tooltip("trust:Q", format=".2f"), "action:N", alt.Tooltip("attack_type:N", title="type")],
    )).properties(width=CHART_WIDTH, height=95)

    return alt.vconcat(top, gap, scores, trust, spacing=8).resolve_scale(
        x="shared", color="independent").configure_view(strokeWidth=0)


def plot_run(result: pd.DataFrame, run_df: pd.DataFrame, labels: Mapping[str, Any],
             upto: pd.Timestamp | None = None, xlim: tuple | None = None) -> plt.Figure:
    """Static matplotlib version of the four panels (for PNG download and figures)."""
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
    ax.set_ylabel("kWh/m²/day")
    ax.legend(loc="upper left", fontsize=8)
    axes[1].plot(r.index, (np.exp(r["gap"]) - 1.0) * 100.0, lw=0.9, color="#37474f")
    axes[1].axhline(0, color="#9e9e9e", lw=0.6)
    axes[1].set_ylabel("gap (%)")
    for (name, col, _, _), color in zip(DETECTORS, DETECTOR_COLORS):
        axes[2].plot(r.index, r[col].clip(upper=2.5), lw=0.9, color=color, label=name)
    axes[2].axhline(1.0, color="#c62828", ls="--", lw=0.8)
    axes[2].set_ylim(0, 2.6)
    axes[2].set_ylabel("scores")
    axes[2].legend(loc="upper left", fontsize=7, ncol=4)
    axes[3].bar(r.index, r["trust"], width=1.0, color=[ACTION_COLORS.get(a, "#9e9e9e") for a in r["action"]])
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


def figure_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------------------ explanations

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-np.clip(x, -50.0, 50.0))))


def explain_day(result: pd.DataFrame, run_df: pd.DataFrame, day: Any, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Plain-language reasons for the decision on one day, strongest first."""
    day = pd.Timestamp(day)
    row = result.loc[day]
    slope = float((cfg.get("fusion") or {}).get("slope") or 8.0)
    reasons: list[tuple[str, str]] = []
    target = float(row["target"])
    gap_pct = float((np.exp(row["gap"]) - 1.0) * 100.0) if pd.notna(row["gap"]) else float("nan")

    if pd.notna(row["z"]):
        z = float(row["z"])
        level = "strong" if abs(z) >= 3 else "medium" if abs(z) >= 2 else "info"
        reasons.append((level, f"The site reports {target:.2f} kWh/m²/day, {gap_pct:+.1f}% away from the "
                               f"{float(row['expected']):.2f} expected from neighbours, clear-sky radiation and "
                               f"weather ({z:+.1f} standard deviations)."))
    nb_cols = [c for c in run_df.columns if c.startswith("nb_")]
    if nb_cols:
        nb = run_df.loc[day, nb_cols].astype(float).dropna()
        if len(nb):
            reasons.append(("info", f"Neighbouring grid cells read {', '.join(f'{v:.2f}' for v in nb)} "
                                    f"kWh/m²/day (mean {nb.mean():.2f})."))
    clr = float(run_df.loc[day, CLRSKY]) if CLRSKY in run_df and pd.notna(run_df.loc[day, CLRSKY]) else None
    clearness = target / clr if clr and clr > 0 else None
    if clr is not None:
        if target > clr:
            reasons.append(("strong", f"The value is above the clear-sky limit ({clr:.2f}): more sunshine than a "
                                      f"cloudless sky can deliver, which is physically impossible."))
        else:
            reasons.append(("info", f"Clear-sky limit {clr:.2f}; the reading is {clearness:.0%} of it."))
    if "cloud" in run_df and pd.notna(run_df.loc[day, "cloud"]):
        cloud = float(run_df.loc[day, "cloud"])
        if cloud >= 0.7 and clearness is not None and clearness >= 0.8:
            reasons.append(("medium", f"Sentinel-2 saw {cloud:.0%} cloud over the site today, yet the reading is "
                                      f"{clearness:.0%} of clear-sky (weak evidence alone: one 10:30 snapshot)."))
        else:
            reasons.append(("info", f"Sentinel-2 passed over today and saw {cloud:.0%} cloud."))
    contrib = []
    for name, col, title, meaning in DETECTORS:
        s = float(row[col])
        contrib.append({"detector": title, "score": s, "p": _sigmoid(slope * (s - 1.0))})
        if s >= 1.0:
            reasons.append(("strong", f"{title} is above its alarm level (score {s:.2f}): {meaning}."))
        elif s >= 0.75:
            reasons.append(("medium", f"{title} is close to its alarm level (score {s:.2f}): {meaning}."))
    if pd.notna(row["est_start"]):
        reasons.append(("info", f"Estimated start of the manipulation: {pd.Timestamp(row['est_start']).date()}."))
    order = {"strong": 0, "medium": 1, "info": 2}
    reasons.sort(key=lambda r: order[r[0]])
    return {"reasons": reasons, "contrib": pd.DataFrame(contrib), "action": str(row["action"]),
            "trust": float(row["trust"]), "p": float(row["p_compromise"]), "type": str(row["attack_type"]),
            "gap_pct": gap_pct}


def contribution_chart(contrib: pd.DataFrame) -> alt.Chart:
    return alt.Chart(contrib).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("p:Q", title="probability contribution", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("detector:N", title=None, sort=None),
        color=alt.Color("p:Q", scale=alt.Scale(domain=[0, 0.5, 1], range=["#22c55e", "#f97316", "#ef4444"]),
                        legend=None),
        tooltip=["detector:N", alt.Tooltip("score:Q", format=".2f"), alt.Tooltip("p:Q", format=".2f")],
    ).properties(height=150)


# ------------------------------------------------------------------ widgets and checks

def badge_html(action: str, day: pd.Timestamp, trust: float) -> str:
    color = ACTION_COLORS.get(action, "#64748b")
    text = ACTION_LABELS.get(action, str(action).upper())
    return (f"<div style='background:{color};color:white;padding:16px;border-radius:14px;text-align:center;"
            f"font-size:32px;font-weight:800;letter-spacing:0.08em;box-shadow:0 0 26px {color}88'>{text}"
            f"<div style='font-size:15px;font-weight:500;letter-spacing:0'>{pd.Timestamp(day).date()} · "
            f"trust {trust:.2f}</div></div>")


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