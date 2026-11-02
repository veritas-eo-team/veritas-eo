# VERITAS: Weather Data Trust Monitor

TSYP14 "Ground Truth Under Attack", Track 2: False-Data Injection.

## What it does

1. Downloads a daily NASA POWER solar-radiation series (plus neighbouring grid cells and other
   variables) and Sentinel-2 cloud statistics, and records every file in a signed, hash-chained log.
2. Predicts each day's value from context an attacker did not touch (neighbours, clear-sky
   radiation, weather variables, season) and monitors the gap between observed and expected.
3. Four detectors watch the gap and the raw series: CUSUM (steps), EWMA + slope test (drifts),
   window matching (replays, stuck sensors) and physics checks (all-sky above clear-sky, sunny
   readings under satellite-observed cloud).
4. Their scores are fused into a daily trust score in [0, 1] and an action:
   accept / deweight / verify / quarantine, with an alert when quarantine starts.
5. A benchmark injects the standard attacks (12% step, drift to 15%, 7-day replay) and smart
   attacks into unseen locations and an unseen year, and reports detection quality, false
   alarms, latency, memory and security overhead.

## Results at a glance

Unseen test locations (5 sites x 20 seeds per attack), test year 2024, after tuning on train
locations only (`results/summary.md` has the full tables):

| Attack | Caught | Median delay | False alarms / location-year | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Step +12% | 100% | 2 days | 0.40 | 0.93 | 0.99 |
| Drift to +15% over 60 days | 100% | 10 days | 0.40 | 0.92 | 0.96 |
| 7-day replay | 100% | 4.5 days | 0.71 | 0.33 | 0.86 |
| Adaptive step (0.9 x detection limit) | 100% | 4.5 days | 0.41 | 0.77 | 0.85 |

Cost: about 0.153 ms per location-day and under 5 MB of peak memory on a laptop.

## Install

Python 3.11, CPU only. Either:

    conda env create -f environment.yml
    conda activate eo-trust

or:

    python -m venv .venv
    .venv\Scripts\activate          (Windows; on Linux/macOS: source .venv/bin/activate)
    pip install -r requirements.lock   (or requirements.txt if the lock file is absent)

## One-command run

    python scripts/run_all.py            # download -> align -> train -> tune -> benchmark -> verify
    python scripts/run_all.py --quick    # quick tuning and benchmark
    python scripts/run_all.py --offline  # no network, cached data only

Finished steps are skipped; `--force` re-runs training, tuning and the benchmark.
Expected runtime on a laptop: about 15 minutes for a full first run (NASA POWER download a few
minutes, benchmark about 2 to 6 minutes), a few minutes with `--quick`.

## Demo

    python scripts/demo.py

Injects a 12% step into a test location and prints the day-by-day trust score and action, then
shows that editing a logged file is caught by the signed log. Falls back to synthetic data if the
real data or model are not present.

Web app (Phase 2): `streamlit run app/app.py`

## Step by step

    python -m src.data.power --all                 # NASA POWER (no key needed)
    python -m src.data.sentinel2 --all             # Sentinel-2 (optional, needs CDSE credentials)
    python -m src.data.align --all                 # clean, aligned frames + quality table
    python -m src.detect.context_model --train     # expected-value model (train locations only)
    python -m src.eval.tune                        # thresholds on train locations -> config/tuned.yaml
    python -m src.eval.run_benchmark --sweep       # results/
    python -m src.pipeline --location kairouan --attack step --size 0.12 --seed 1
    python -m src.trust.verify --allow-missing     # check the provenance log
    pytest                                         # offline tests (synthetic data)

## Expected outputs

- `data/clean/<location>.parquet`: aligned daily frames
- `models/context_model.joblib`, `models/fusion.joblib`
- `config/tuned.yaml`: tuned thresholds and a record of the tuning run
- `results/benchmark_table.csv`, `per_scenario.csv`, `summary.md`, `system_info.json`,
  `catch_vs_size.png`, `delay_vs_drift_duration.png`, `false_alarms_vs_threshold.png`, `reliability.png`
- `data/provenance/log.jsonl`: signed hash-chained record of every download and derived file


## Tested on

Intel x86-64 laptop CPU (8 physical / 12 logical cores), 31.7 GB RAM, Windows (build 10.0.22621),
Python 3.11.18, NumPy 2.4.6, pandas 3.1.0; no GPU. Details in `results/system_info.json`.


## Sentinel-2 credentials (optional)

Create a free Copernicus Data Space account and an OAuth client in the Sentinel Hub dashboard
(User settings -> OAuth clients), then set them for the current terminal only:

    $env:CDSE_CLIENT_ID="..."          (PowerShell; bash: export CDSE_CLIENT_ID=...)
    $env:CDSE_CLIENT_SECRET="..."

Never write credentials into a file in the repo. Without them the system still runs; only the
cloud-consistency check is inactive.

## Provenance log

Every download and derived file is hashed, chained to the previous record and signed with
Ed25519. To check it on a fresh clone (raw data is not in the repo):

   python -m src.trust.verify --allow-missing --trust 21ded33c4cec6bf2 --expected-head 038ee7c91d72c20521f0b32035a0cf426e98a79486253f8daa9c01d8370ec7af

`--trust` pins the signing key id of the reference run; the log head hash is printed in
`results/summary.md`. Only one machine should append to the log (a second machine would create
its own key).

## Repository map

    config/          settings.yaml (all settings), tuned.yaml (written by tuning)
    src/common.py    paths, config loading, logging
    src/data/        power.py (NASA POWER), sentinel2.py (Sentinel-2), align.py (clean frames)
    src/trust/       provenance.py (signed log), verify.py (verification, tamper demo, overhead)
    src/attacks/     inject.py (step, drift, replay, spike, stuck, adaptive, slow ramp)
    src/detect/      context_model.py, step.py, drift.py, replay.py, consistency.py, fusion.py
    src/respond/     policy.py (trust -> action, hysteresis)
    src/pipeline.py  end-to-end daily pipeline
    src/eval/        scenarios.py, metrics.py, run_benchmark.py, tune.py
    scripts/         run_all.py, demo.py, anon_check.py
    tests/           pytest suite (synthetic data, no internet)
    results/         benchmark outputs

## Reproducibility

- Every random choice takes a seed; attack starts are drawn per seed inside February-September
  of the test year.
- Split by location and year: 10 train locations, 5 test locations; the context model and all
  thresholds use train locations and years before 2024 only.
- Detectors are causal: day t uses data up to day t only (tested).

## Known limitations

- Signatures prove that files are unchanged since download, not that the source was honest.
- An attacker who also changes the neighbouring cells and the other variables consistently is out
  of scope: the context model trusts them.
- Per-site calibration uses the first year of a series; an attack already active then is absorbed.
- Drifts are caught later than steps (median 10 days); very slow ramps can stay below the
  thresholds for longer.
- Replay alarms persist for a while after a short replay ends, which lowers per-day precision.
- The clean-run false-alarm estimate on test locations rests on few location-years; the
  pre-attack days of all runs give a larger sample (0.4-0.7 per location-year).
- Cloudy tropical sites are slow: at the test site abidjan, a 12% step takes a median of about a month to alarm, and a 2.7% step about five months.
- The clear-sky check only catches inflated values; downward bias relies on the statistical detectors and is caught later.