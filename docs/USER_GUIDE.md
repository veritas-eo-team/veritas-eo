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

## 3. Web console

The console lives in `web/` and talks to the Python pipeline through a local API. First run:

    cd web
    npm install
    npm run dev        # API + console on http://127.0.0.1:3000

`npm run dev` starts `scripts/api_server.py` (with `.venv/Scripts/python.exe` when it exists, otherwise `python`) and then Next.js on 127.0.0.1:3000. Separately: `python scripts/api_server.py` and, in `web/`, `npm run dev:web`.

| Page | What it shows |
|---|---|
| Overview | system status (clean data, model, tuned settings, log check), the pipeline, headline results and published failure modes |
| Attack lab | choose location, attack (none/step/drift/replay/spike/stuck), size, duration, start, noise, seed; interactive chart of observed vs expected, gap, detector scores and trust; a per-day explanation; choices are shown as a judge challenge mode |
| Live replay | the same run revealed day by day, with the action badge, an alert feed and a progress bar |
| Integrity | provenance log with a per-record check, "Verify chain", tamper demo, gateway cases, overhead measurement |
| Benchmark | benchmark tables, figures and the full summary |
| Assess your data | upload a daily CSV, give coordinates, score it and download the verdict |

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

    python -m src.trust.verify --allow-missing --trust 21ded33c4cec6bf2 --expected-head f5368050c207f622701fdeffd32e683589580b87995c95f1dbb71a8c482bc4e0
    python -m src.trust.verify --demo            # tamper demo on a temporary copy
    python -m src.trust.verify --bench           # signing / verification overhead
    python -m src.trust.gateway --demo           # signed sensor + gateway attack cases
    python -m src.trust.gateway --bench          # per-reading overhead

The gateway demo shows altered, replayed, forged, unregistered and delayed messages rejected, and a compromised sensor with a valid key accepted. That last case is why the statistical detectors exist.

## 7. Tests

    pytest -q

318 offline tests on synthetic data, about 1–2 minutes.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `pytest` "not recognized" | run `conda activate eo-trust` first |
| `npm` "not recognized" | install Node.js 20 or later, then run `npm install` in `web/` |
| "run: python -m src.detect.context_model --train" | the model file is missing; run that command |
| No Sentinel-2 data | optional; without it the cloud check is inactive and everything else works |
| Verify fails on missing files | add `--allow-missing` (raw data is not in the repository) |