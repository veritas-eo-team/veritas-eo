"""Attack lab: inject an attack, run the pipeline, explore every signal, explain any flagged day."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ui  # noqa: E402

ui.page_setup("Attack lab")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

ui.hero("Attack lab", "Inject an attack, watch every detector react, and ask why a day was flagged")
controls = ui.scenario_controls("lab")
try:
    result, labels, run_df, label, demo = ui.run_scenario(**controls)
except ValueError as exc:
    st.error(f"This attack cannot be placed there: {exc}")
    st.stop()

ui.demo_banner(demo)
st.caption(f"{label} · attack: {controls['attack']} · the detector is never told when or whether an attack happens")

s = ui.summarize_run(result, labels)
c = st.columns(5)
ui.card(c[0], "True start", "-" if s["true_start"] is None else str(s["true_start"].date()),
        "hidden from the detector")
ui.card(c[1], "First alarm", "-" if s["first_alarm"] is None else str(s["first_alarm"].date()),
        "verify or quarantine", "ok" if s["first_alarm"] is not None else ("mid" if s["true_start"] is None else "bad"))
ui.card(c[2], "Delay", "-" if s["delay"] is None else f"{s['delay']} days")
ui.card(c[3], "Detected type", s["type"] or ("no attack" if s["true_start"] is None else "-"))
ui.card(c[4], "Lowest trust", f"{s['lowest_trust']:.2f}", "after the attack start" if s["true_start"] is not None else "")
if s["true_start"] is None:
    st.info(f"Clean run: {s['alarm_days']} alarm days (verify or quarantine) in the whole window.")

st.markdown("#### Signals")
st.altair_chart(ui.altair_run(result, run_df, labels))
st.caption("Scroll to zoom, drag to pan, double-click to reset, hover for values. Red band: attacked days. "
           "Scores alarm at 1.0. Trust colours: green accept, yellow de-weight, orange verify, red quarantine.")
st.download_button("Download chart (PNG)", ui.figure_png(ui.plot_run(result, run_df, labels)),
                   file_name=f"veritas_{controls['location']}_{controls['attack']}.png", mime="image/png")

st.markdown("#### Why was this day flagged?")
flagged = result[result["action"] != "accept"]
if flagged.empty:
    st.success("No day in this run was flagged: every value was accepted.")
else:
    days = list(flagged.index)
    default = 0
    if s["first_alarm"] is not None and s["first_alarm"] in days:
        default = days.index(s["first_alarm"])
    day = st.selectbox("Flagged day", days, index=default,
                       format_func=lambda d: f"{d.date()} · {result.loc[d, 'action']} · trust {result.loc[d, 'trust']:.2f}")
    ex = ui.explain_day(result, run_df, day, ui.get_cfg())
    left, right = st.columns([1, 2])
    with left:
        st.markdown(ui.badge_html(ex["action"], day, ex["trust"]), unsafe_allow_html=True)
        st.markdown(f"<div class='vx-sub' style='margin-top:10px'>compromise probability {ex['p']:.2f} · "
                    f"type <b>{ex['type']}</b></div>", unsafe_allow_html=True)
        st.altair_chart(ui.contribution_chart(ex["contrib"]))
    with right:
        for level, text in ex["reasons"]:
            st.markdown(f"<div class='vx-reason {level}'>{text}</div>", unsafe_allow_html=True)

with st.expander(f"All {len(flagged)} days not accepted"):
    view = pd.DataFrame({
        "observed": flagged["target"].round(2),
        "expected": flagged["expected"].round(2),
        "gap (%)": ((np.exp(flagged["gap"]) - 1.0) * 100).round(1),
        "trust": flagged["trust"].round(2),
        "action": flagged["action"],
        "attack type": flagged["attack_type"],
        "estimated start": flagged["est_start"].dt.date,
    }, index=flagged.index.date)
    st.dataframe(view, width="stretch")

out = result.copy()
out["attacked"] = False
if labels.get("start") is not None:
    out["attacked"] = (out.index >= labels["start"]) & (out.index <= labels["end"])
st.download_button("Download this run (CSV)", out.to_csv(date_format="%Y-%m-%d").encode("utf-8"),
                   file_name=f"veritas_{controls['location']}_{controls['attack']}.csv", mime="text/csv")