"""Live replay: days arrive one by one; the badge shows the decision for each new day."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ui  # noqa: E402

ui.page_setup("Live replay")

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

ui.hero("Live replay", "Days arrive one at a time, as they would every morning in operation")
controls = ui.scenario_controls("live")
speed = st.sidebar.slider("Speed (days per second)", 1, 60, 15, key="live_speed")
st.caption("The pipeline is causal (tested): each day's decision uses only data up to that day, so revealing "
           "a run day by day is the same as running it every morning.")

if st.button("▶ Start replay", type="primary"):
    try:
        result, labels, run_df, label, demo = ui.run_scenario(**controls)
    except ValueError as exc:
        st.error(f"This attack cannot be placed there: {exc}")
        st.stop()
    ui.demo_banner(demo)
    start, end = labels.get("start"), labels.get("end")
    last = result.index[-1]
    if start is None:
        view_start, view_end = result.index[max(0, len(result) - 180)], last
    else:
        view_start = max(result.index[0], start - pd.Timedelta(days=45))
        long_attack = controls["attack"] in ("step", "drift")
        view_end = min(last, start + pd.Timedelta(days=150) if long_attack else end + pd.Timedelta(days=60))
    days = result.loc[view_start:view_end].index
    st.write(f"{label}: replaying {days[0].date()} to {days[-1].date()}"
             + ("" if start is None else f" · attack starts {start.date()} (hidden from the detector)"))

    top = st.columns([1, 2])
    badge, alert = top[0].empty(), top[1].empty()
    chart = st.empty()
    redraw_every = max(1, speed // 6)
    for i, day in enumerate(days):
        if i % redraw_every == 0 or i == len(days) - 1:
            row = result.loc[day]
            badge.markdown(ui.badge_html(row["action"], day, row["trust"]), unsafe_allow_html=True)
            alerts = result.loc[view_start:day]
            alerts = alerts[alerts["alert"]]
            if len(alerts):
                a_day = alerts.index[-1]
                alert.error(f"🚨 ALERT {a_day.date()}: quarantine started · detected type "
                            f"'{alerts.loc[a_day, 'attack_type']}'")
            else:
                alert.info("Monitoring… no alert so far.")
            chart.altair_chart(ui.altair_run(result, run_df, labels, upto=day, xlim=(view_start, view_end)))
        time.sleep(1.0 / speed)
    s = ui.summarize_run(result, labels)
    if s["first_alarm"] is not None:
        st.success(f"Detected {s['delay']} day(s) after the attack began · type '{s['type']}'.")
    elif start is not None:
        st.warning("No alarm after the attack start in this run.")
else:
    st.info("Choose a scenario in the sidebar and press Start replay.")