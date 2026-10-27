"""Benchmark: tables, summary and charts written by python -m src.eval.run_benchmark."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ui  # noqa: E402

ui.page_setup("Benchmark")

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.title("Benchmark")
res = ui.results_dir()
table_path = res / "benchmark_table.csv"
if not table_path.exists():
    st.info("No results yet. Run `python -m src.eval.run_benchmark --sweep` (about 7 minutes on a laptop).")
    st.stop()

table = pd.read_csv(table_path)
for group, title in (("test", "Unseen test locations, 2024"), ("unseen_year", "Unseen year (train locations, 2024)")):
    part = table[table["group"] == group].drop(columns=["group"])
    if len(part):
        st.subheader(title)
        st.dataframe(part.round(3), hide_index=True, width="stretch")

charts = [("catch_vs_size.png", "Catch rate vs step size"),
          ("delay_vs_drift_duration.png", "Delay vs drift duration"),
          ("false_alarms_vs_threshold.png", "False alarms and catch rate vs threshold"),
          ("reliability.png", "Reliability of the compromise probability"),
          ("attack_examples.png", "The attacks")]
existing = [(f, t) for f, t in charts if (res / f).exists()]
if existing:
    st.subheader("Charts")
    cols = st.columns(2)
    for i, (fname, title) in enumerate(existing):
        with cols[i % 2]:
            st.image(str(res / fname), caption=title, width="stretch")

summary = res / "summary.md"
if summary.exists():
    with st.expander("Full summary (results/summary.md)"):
        st.markdown(summary.read_text(encoding="utf-8"))
