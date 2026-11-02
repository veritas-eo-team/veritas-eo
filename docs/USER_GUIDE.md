# User guide

## 1. Set up (once)

Python 3.11, CPU only.

    conda env create -f environment.yml
    conda activate eo-trust

Every new terminal: `conda activate eo-trust` (the prompt must show `(eo-trust)`). The clean data and trained models are in the repository, so everything below works offline.

## 2. Thirty-second demo

    python scripts/demo.py                       # 12% step on a test location
    python scripts/demo.py --location lyon --fast

Prints the day-by-day trust score and action as the attack starts, then shows that editing a logged file is caught by the signed log (on a temporary copy).

## 3. Web app

    streamlit run app/app.py

| Page | What it shows |
|---|---|
| Home | system status (clean data, model, tuned settings, log check) and the headline benchmark results |
| Attack lab | choose location, attack (none/step/drift/replay/spike/stuck), size, duration, start, noise, seed; interactive chart of observed vs expected, gap, detector scores and trust; "Why was this day flagged?"; CSV and PNG download |
| Live replay | the same run revealed day by day, with the action badge and an alert when quarantine starts |
| Integrity | provenance log with a per-record check, "Verify now", tamper demo, gateway cases, overhead measurement |
| Benchmark | benchmark tables, charts and summary |

If the data or model are missing, the app falls back to synthetic data and shows a banner.

## 4. Run one attack scenario

    python -m src.pipeline --location kairouan --attack step --size 0.12 --seed 1
    python -m src.pipeline --location abidjan --attack drift --size 0.15 --duration 60
    python -m src.pipeline --location lyon --attack replay --duration 7 --noise 0.01

Options: `--attack step|drift|replay|spike|stuck|none`, `--size`, `--duration` (days), `--start YYYY-MM-DD` (default: a seeded day in February–September of the test year), `--noise`, `--seed`.

It prints the true and detected start, the delay, the attack type, false alarms before the start, which detectors fired, the action counts and the lowest trust, and saves `results/run_<location>_<attack>.csv`. These run files are for exploration; there is no need to commit them.

**Reading the output (one row per day)**

| Column | Meaning |
|---|---|
| `target`, `expected` | observed value and the context model's prediction |
| `gap`, `z` | log(observed/expected) and the gap in units of uncertainty |
| `step_score`, `drift_score`, `replay_score`, `consistency_score` | detector scores; ≥ 1 means that detector alarms |
| `p_compromise`, `trust` | fused probability of tampering and trust = 1 − p |
| `attack_type` | none / step / drift / replay / stuck |
| `est_start` | estimated start of the current episode |
| `action` | accept (trust ≥ 0.8), deweight (≥ 0.5), verify (≥ 0.2), quarantine (< 0.2, or exact replay / stuck) |
| `alert` | True on the first day of each quarantine episode |

## 5. Reproduce the results

    python -m src.eval.run_benchmark --sweep     # 2 to 6 minutes; writes results/
    python -m src.eval.tune                      # thresholds on train locations -> config/tuned.yaml
    python -m src.detect.context_model --train   # retrain the expected-value model

The benchmark is deterministic: a rerun reproduces `results/benchmark_table.csv` exactly. Outputs:

- `results/summary.md`: all tables (standard attacks, unseen year, sweeps, adaptive attacker, cost, system)
- `results/benchmark_table.csv`, `results/per_scenario.csv` (one row per run)
- `results/catch_vs_size.png`, `delay_vs_drift_duration.png`, `false_alarms_vs_threshold.png`, `reliability.png`

Full pipeline from scratch (download → align → train → tune → benchmark → verify):

    python scripts/run_all.py [--offline] [--quick] [--force]

## 6. Trust anchor

    python -m src.trust.verify --allow-missing --trust 21ded33c4cec6bf2 --expected-head 038ee7c91d72c20521f0b32035a0cf426e98a79486253f8daa9c01d8370ec7af
    python -m src.trust.verify --demo            # tamper demo on a temporary copy
    python -m src.trust.verify --bench           # signing / verification overhead
    python -m src.trust.gateway --demo           # signed sensor + gateway attack cases
    python -m src.trust.gateway --bench          # per-reading overhead

The gateway demo shows altered, replayed, forged, unregistered and delayed messages rejected, and a compromised sensor with a valid key accepted. That last case is why the statistical detectors exist.

## 7. Tests

    pytest -q

267 offline tests on synthetic data, about 1–2 minutes.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `pytest` / `streamlit` "not recognized" | run `conda activate eo-trust` first |
| "run: python -m src.detect.context_model --train" | the model file is missing; run that command |
| No Sentinel-2 data | optional; without it the cloud check is inactive and everything else works |
| Verify fails on missing files | add `--allow-missing` (raw data is not in the repository) |