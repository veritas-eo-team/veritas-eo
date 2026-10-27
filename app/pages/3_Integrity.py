"""Integrity: the signed hash-chained log, verification, tamper demo, gateway demo, overhead."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ui  # noqa: E402

ui.page_setup("Integrity")

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.trust.provenance import load_public_keys, read_log  # noqa: E402

st.title("Integrity")
st.markdown("Every downloaded and derived file is hashed, chained to the previous record and signed. "
            "Editing, deleting or reordering anything breaks the chain.")


def log_table(records, checks, mark_seq=None) -> pd.DataFrame:
    rows = []
    for rec, ok in zip(records, checks):
        status = "âœ…" if ok else "âŒ"
        if mark_seq is not None and rec.get("seq") == mark_seq:
            status = "âŒ tampered (temporary copy)"
        rows.append({"seq": rec.get("seq"), "kind": rec.get("kind"), "file": rec.get("path") or "-",
                     "created (UTC)": rec.get("created_at"), "file hash": str(rec.get("sha256", ""))[:12],
                     "record hash": str(rec.get("record_hash", ""))[:12], "key id": rec.get("public_key_id"),
                     "signature": status})
    return pd.DataFrame(rows)


try:
    records = read_log()
except Exception as exc:
    records = []
    st.error(f"Cannot read the log: {exc}")
keys = load_public_keys()
checks = ui.record_checks(records, keys) if records else []

st.subheader("Provenance log")
if records:
    st.caption(f"{len(records)} records; signing key ids present: {', '.join(sorted(keys)) or 'none'}")
    st.dataframe(log_table(records, checks), hide_index=True, width="stretch")
else:
    st.info("No provenance records yet.")

c1, c2 = st.columns(2)
if c1.button("Verify now"):
    from src.trust.verify import verify_log
    res = verify_log(check_files=True, allow_missing=True)
    if res.ok:
        st.success(f"Chain, hashes and signatures OK: {res.n_records} records. Head hash `{res.head_hash}`")
    else:
        st.error(f"Verification failed at record {res.first_bad_seq}: {res.reason}")
if c2.button("Tamper demo"):
    from src.trust.verify import tamper_demo
    res = tamper_demo()
    if not res.ok:
        st.error(f"Tampering detected at record {res.first_bad_seq}: {res.reason}")
        if records:
            st.dataframe(log_table(records, checks, mark_seq=res.first_bad_seq), hide_index=True,
                         width="stretch")
        st.caption("The edit was made on a temporary copy; the real data and log are untouched.")
    else:
        st.warning("The tamper demo was not caught; check the log.")

st.subheader("Signed sensor and gateway")
st.markdown("Each reading is signed by the sensor and checked by the gateway: device, signature, "
            "timestamp window, sequence number and nonce.")
if st.button("Run the gateway cases"):
    from src.trust.gateway import demo_attacks
    cases = demo_attacks()
    view = cases[["case", "expected", "accepted", "reason"]].copy()
    view["accepted"] = view["accepted"].map({True: "accepted", False: "rejected"})
    st.dataframe(view, hide_index=True, width="stretch")
    st.info("The compromised sensor's biased reading passes every cryptographic check. "
            "Only the statistical detectors can catch it, which is why they exist.")

st.subheader("Security overhead")
if st.button("Measure overhead"):
    from src.trust.gateway import bench_gateway
    from src.trust.verify import bench_overhead
    b = bench_overhead(n=300)
    g = bench_gateway(n=100)
    st.dataframe(pd.DataFrame([
        {"operation": "sign one provenance record", "value": f"{b['sign_ms']:.3f} ms"},
        {"operation": "verify one provenance record (chain)", "value": f"{b['chain_verify_ms_per_record']:.3f} ms"},
        {"operation": "hash 1 MB", "value": f"{b['hash_ms_per_mb']:.3f} ms"},
        {"operation": "provenance record size", "value": f"{b['bytes_per_record']:.0f} bytes"},
        {"operation": "sign one reading (sensor)", "value": f"{g['sign_ms']:.3f} ms"},
        {"operation": "check one reading (gateway)", "value": f"{g['check_ms']:.3f} ms"},
        {"operation": "check + signed log append", "value": f"{g['check_and_log_ms']:.3f} ms"},
    ]), hide_index=True, width="stretch")
