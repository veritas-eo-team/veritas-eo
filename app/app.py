"""Weather Data Trust Monitor: overview page.  Run with: streamlit run app/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui  # noqa: E402

ui.page_setup("Overview")

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.common import PATHS  # noqa: E402

st.title(ui.TITLE)
st.caption("Detecting false-data injection in a daily solar-radiation series (NASA POWER + Sentinel-2)")

st.markdown("""
1. **Signed data.** Every downloaded file and every sensor reading is hashed, chained and signed (Ed25519).
2. **Context model.** Each day's value is predicted from what the attacker cannot touch: neighbouring grid cells, clear-sky radiation, weather variables, season.
3. **Four detectors** watch the gap between observed and expected: CUSUM (steps), EWMA and slope test (drifts), window matching (replays, stuck sensors), physics checks (above clear-sky, sunny under satellite-observed cloud).
4. **Trust score.** The detector scores are fused into a daily trust score between 0 and 1.
5. **Action.** Accept, de-weight, verify or quarantine, with an alert when quarantine starts.
""")

st.subheader("System status")
cols = st.columns(4)
n_clean = len(list(PATHS["clean"].glob("*.parquet")))
model, model_err = ui.get_model()
cols[0].metric("Clean data", f"{n_clean} locations", "ready" if n_clean else "missing",
               delta_color="normal" if n_clean else "inverse")
cols[1].metric("Context model", "trained" if model is not None else "missing",
               "ready" if model is not None else "demo mode", delta_color="normal" if model is not None else "inverse")
cols[2].metric("Tuned settings", "loaded" if PATHS["tuned"].exists() else "defaults",
               "config/tuned.yaml" if PATHS["tuned"].exists() else "not tuned")


@st.cache_data(ttl=60)
def log_status() -> dict:
    try:
        from src.trust.verify import verify_log
        res = verify_log(check_files=False)
        return {"ok": bool(res.ok), "n": int(res.n_records), "head": res.head_hash, "reason": res.reason}
    except Exception as exc:
        return {"ok": False, "n": 0, "head": None, "reason": str(exc)}


status = log_status()
cols[3].metric("Provenance log", f"{status['n']} records", "chain and signatures OK" if status["ok"] else "check failed",
               delta_color="normal" if status["ok"] else "inverse")
if not status["ok"] and status["reason"]:
    st.caption(f"Log check: {status['reason']}")
if model is None:
    ui.demo_banner(True)

st.subheader("Headline results (unseen test locations, 2024)")
table_path = ui.results_dir() / "benchmark_table.csv"
if table_path.exists():
    table = pd.read_csv(table_path)
    main = table[table["group"] == "test"].copy()
    view = pd.DataFrame({
        "attack": main["attack"],
        "runs": main["n_scenarios"],
        "caught": main["catch_rate"].map(lambda v: "-" if pd.isna(v) else f"{v:.0%}"),
        "median delay (days)": main["median_delay_days"],
        "false alarms / location-year": main["fa_per_location_year"].round(2),
        "F1": main["f1"].round(2),
        "ROC-AUC": main["roc_auc"].round(2),
    })
    st.dataframe(view, hide_index=True, width="stretch")
else:
    st.info("No benchmark results yet: run `python -m src.eval.run_benchmark --sweep`.")

st.markdown("Use the pages in the sidebar: **Attack lab**, **Live replay**, **Integrity**, **Benchmark**.")
