"""Upload your data: score any daily CSV with VERITAS (any-file mode)."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ui  # noqa: E402

ui.page_setup("Upload your data")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import score_file as SF  # noqa: E402
from src.data import power  # noqa: E402

UNITS = {"mj": "MJ/m²/day (NASA POWER)", "kwh": "kWh/m²/day", "wm2": "daily mean W/m²"}

ui.hero("Upload your data", "Bring any daily solar series, from anywhere, and let VERITAS check it")
st.markdown("Upload a CSV with a **date** column and a **daily value** column. VERITAS downloads the "
            "neighbouring cells and weather context for your coordinates from NASA POWER, learns the "
            "site's normal behaviour from the **first year**, then checks every day after it. "
            "The detector is never told whether your file contains an attack.")


@st.cache_data(show_spinner="Scoring the file (the first run downloads context from NASA POWER)...")
def score_bytes(data: bytes, lat: float, lon: float, units: str, date_col: str, value_col: str, name: str):
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        target, extras = SF.read_input(path, date_col or None, value_col or None, units)
        warning = target.attrs.get("unit_warning")
        result, info = SF.score_series(target, ui.get_cfg(), ui.get_model()[0], extras=extras,
                                       lat=lat, lon=lon, name=name)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    frame = info.pop("frame")
    return result, frame, info, warning


@st.cache_data
def demo_csv() -> bytes:
    """Kairouan 2023-2024 with a hidden +12% step from 1 May 2024; context columns included (works offline)."""
    df = ui.load_frame("kairouan").loc["2023-01-01":"2024-12-31"].copy()
    t = df["target"].copy()
    t.loc["2024-05-01":] *= 1.12
    out = df[[c for c in df.columns if c.startswith(("nb_", "ctx_"))]].copy()
    out.insert(0, "value", t.to_numpy())
    out.insert(0, "date", df.index.strftime("%Y-%m-%d"))
    return out.to_csv(index=False).encode("utf-8")


# ------------------------------------------------------------------ inputs

left, right = st.columns([3, 2])
with left:
    up = st.file_uploader("Daily CSV file", type=["csv", "txt"])
    st.caption("One row per day. Common column names (date, value, ghi, radiation...) are detected automatically.")
with right:
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude", -90.0, 90.0, 35.68, 0.01, format="%.2f")
    lon = c2.number_input("Longitude", -180.0, 180.0, 10.10, 0.01, format="%.2f")
    units = st.selectbox("Units of your values", list(UNITS), format_func=UNITS.get)

with st.expander("Advanced: column names, and a known attack start"):
    a1, a2 = st.columns(2)
    date_col = a1.text_input("Date column (blank = auto)")
    value_col = a2.text_input("Value column (blank = auto)")
    known = st.checkbox("I know when an attack starts in this file (to measure the delay)")
    known_start = st.date_input("True attack start", value=pd.Timestamp("2024-05-01")) if known else None

b1, b2 = st.columns([1, 2])
run = b1.button("Score this file", type="primary", disabled=up is None)
demo_ok = "kairouan" in ui.list_locs()
use_demo = b2.button("Try the demo file (Kairouan, hidden +12% attack, works offline)", disabled=not demo_ok)

if run and up is not None:
    st.session_state["upload_job"] = {
        "data": up.getvalue(), "name": Path(up.name).stem, "lat": float(lat), "lon": float(lon),
        "units": units, "date_col": date_col.strip(), "value_col": value_col.strip(),
        "known": pd.Timestamp(known_start) if known_start is not None else None}
if use_demo:
    st.session_state["upload_job"] = {
        "data": demo_csv(), "name": "demo_kairouan_attacked", "lat": 35.68, "lon": 10.10, "units": "mj",
        "date_col": "", "value_col": "", "known": pd.Timestamp("2024-05-01")}

job = st.session_state.get("upload_job")
if job is None:
    st.info("Upload a file or try the demo to see a result.")
    st.stop()

try:
    result, frame, info, warning = score_bytes(job["data"], job["lat"], job["lon"], job["units"],
                                               job["date_col"], job["value_col"], job["name"])
except SF.InputError as exc:
    st.error(f"This file cannot be scored: {exc}")
    st.stop()
except power.PowerAPIError as exc:
    st.error(f"NASA POWER could not be reached ({exc}). Check the internet connection, "
             "or include nb_* and ctx_* columns in the file to work offline.")
    st.stop()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

# ------------------------------------------------------------------ results

if warning:
    st.warning(warning)
if info["short_reference"]:
    st.warning("Less than about a year of data: the reference period is short, so results may be unreliable.")
ref_end = min(result.index[0] + pd.Timedelta(days=info["ref_days"] - 1), result.index[-1])
st.caption(f"{info['name']} · {info['days']} days ({info['valid_days']} with values) · context: {info['mode']} · "
           f"{len(info['neighbours'])} neighbours · reference period {result.index[0].date()} to {ref_end.date()} "
           f"(should be clean)")

eps = SF.alarm_episodes(result)
alarm = result["action"].isin(ui.ALARM_ACTIONS)
first = alarm[alarm].index[0] if alarm.any() else None
labels = {"start": None, "end": None}
if job["known"] is not None:
    labels = {"start": job["known"], "end": result.index[-1]}

c = st.columns(5)
ui.card(c[0], "Verdict", "Tampering suspected" if eps else "Looks genuine",
        f"{len(eps)} alarm episode(s)", "bad" if eps else "ok")
ui.card(c[1], "First alarm", "-" if first is None else str(first.date()), "verify or quarantine")
est = eps[0]["est_start"] if eps else None
ui.card(c[2], "Likely start", "-" if est is None or pd.isna(est) else str(pd.Timestamp(est).date()),
        "estimated by VERITAS")
ui.card(c[3], "Quarantined days", str(int((result["action"] == "quarantine").sum())), f"of {len(result)}")
if labels["start"] is not None:
    after = alarm[result.index >= labels["start"]]
    delay = int((after[after].index[0] - labels["start"]).days) if after.any() else None
    ui.card(c[4], "Delay", "missed" if delay is None else f"{delay} days",
            f"true start {labels['start'].date()}", "bad" if delay is None else "ok")
else:
    ui.card(c[4], "Lowest trust", f"{result['trust'].min():.2f}")

if eps:
    st.markdown("#### Alarm episodes")
    st.dataframe(pd.DataFrame([{
        "from": e["start"].date(), "to": e["end"].date(), "days": e["days"], "type": e["type"],
        "likely start": "" if pd.isna(e["est_start"]) else pd.Timestamp(e["est_start"]).date(),
    } for e in eps]), width="stretch", hide_index=True)
else:
    st.success("No alarms: the series agrees with its neighbours, physics and context.")

st.markdown("#### Signals")
st.altair_chart(ui.altair_run(result, frame, labels))
st.caption("Scroll to zoom, drag to pan, double-click to reset, hover for values. "
           "Red band: the attack period you entered (if any). Scores alarm at 1.0.")

st.markdown("#### Why was this day flagged?")
flagged = result[result["action"] != "accept"]
if flagged.empty:
    st.success("No day was flagged: every value was accepted.")
else:
    days = list(flagged.index)
    default = days.index(first) if first is not None and first in days else 0
    day = st.selectbox("Flagged day", days, index=default,
                       format_func=lambda d: f"{d.date()} · {result.loc[d, 'action']} · trust {result.loc[d, 'trust']:.2f}")
    ex = ui.explain_day(result, frame, day, ui.get_cfg())
    l, r = st.columns([1, 2])
    with l:
        st.markdown(ui.badge_html(ex["action"], day, ex["trust"]), unsafe_allow_html=True)
        st.markdown(f"<div class='vx-sub' style='margin-top:10px'>compromise probability {ex['p']:.2f} · "
                    f"type <b>{ex['type']}</b></div>", unsafe_allow_html=True)
        st.altair_chart(ui.contribution_chart(ex["contrib"]))
    with r:
        for level, text in ex["reasons"]:
            st.markdown(f"<div class='vx-reason {level}'>{text}</div>", unsafe_allow_html=True)

st.download_button("Download the verdict for every day (CSV)",
                   result.to_csv(date_format="%Y-%m-%d", index_label="date").encode("utf-8"),
                   file_name=f"veritas_score_{job['name']}.csv", mime="text/csv")