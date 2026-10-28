"""VERITAS: overview page.  Run with: streamlit run app/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui  # noqa: E402

ui.page_setup("Overview")

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.common import PATHS  # noqa: E402

ui.hero(ui.BRAND, "Weather data trust monitor · detects false-data injection in a daily solar-radiation "
                  "series using NASA POWER and Sentinel-2")

steps = [
    ("🔏", "Signed data", "every file and reading hashed, chained and signed"),
    ("🛰️", "Context model", "value expected from neighbours, clear-sky and weather"),
    ("📈", "Four detectors", "steps, drifts, replays and physics checks"),
    ("⚖️", "Trust score", "detector scores fused into a daily 0-1 score"),
    ("🚦", "Action", "accept · de-weight · verify · quarantine"),
]
for col, (icon, title, text) in zip(st.columns(5), steps):
    col.markdown(f"<div class='vx-card'><div style='font-size:1.6rem'>{icon}</div>"
                 f"<div class='value' style='font-size:1.05rem'>{title}</div><div class='note'>{text}</div></div>",
                 unsafe_allow_html=True)

st.markdown("### System status")
cols = st.columns(4)
n_clean = len(list(PATHS["clean"].glob("*.parquet")))
model, _ = ui.get_model()
ui.card(cols[0], "Clean data", f"{n_clean} locations", "ready" if n_clean else "missing", "ok" if n_clean else "bad")
ui.card(cols[1], "Context model", "trained" if model is not None else "missing",
        "ready" if model is not None else "demo mode", "ok" if model is not None else "bad")
tuned = PATHS["tuned"].exists()
ui.card(cols[2], "Settings", "tuned" if tuned else "defaults", "config/tuned.yaml" if tuned else "not tuned yet",
        "ok" if tuned else "mid")


@st.cache_data(ttl=60)
def log_status() -> dict:
    try:
        from src.trust.verify import verify_log
        res = verify_log(check_files=False)
        return {"ok": bool(res.ok), "n": int(res.n_records), "reason": res.reason}
    except Exception as exc:
        return {"ok": False, "n": 0, "reason": str(exc)}


status = log_status()
ui.card(cols[3], "Provenance log", f"{status['n']} records",
        "chain and signatures OK" if status["ok"] else "check failed", "ok" if status["ok"] else "bad")
ui.demo_banner(model is None)

st.markdown("### Headline results · unseen test locations, 2024")
table_path = ui.results_dir() / "benchmark_table.csv"
if table_path.exists():
    table = pd.read_csv(table_path)
    main = table[table["group"] == "test"].set_index("attack")
    attacks = [a for a in ("step", "drift", "replay") if a in main.index]
    cols = st.columns(5)
    caught = main.loc[attacks, "catch_rate"].min() if attacks else float("nan")
    ui.card(cols[0], "Standard attacks caught", f"{caught:.0%}" if caught == caught else "-",
            f"{int(main.loc[attacks, 'n_scenarios'].sum())} runs", "ok")
    for col, attack in zip(cols[1:4], attacks):
        ui.card(col, f"Median delay · {attack}", f"{main.loc[attack, 'median_delay_days']:g} days",
                f"F1 {main.loc[attack, 'f1']:.2f} · AUC {main.loc[attack, 'roc_auc']:.2f}")
    fa = main.loc[attacks, "fa_per_location_year"].mean() if attacks else float("nan")
    ui.card(cols[4], "False alarms", f"{fa:.2f}", "per location-year (target ≤ 1)", "ok" if fa <= 1 else "bad")
    with st.expander("Full table"):
        st.dataframe(main.reset_index().drop(columns=["group"]).round(3), hide_index=True, width="stretch")
else:
    st.info("No benchmark results yet: run `python -m src.eval.run_benchmark --sweep`.")

st.caption("Pages in the sidebar: Attack lab · Live replay · Integrity · Benchmark")