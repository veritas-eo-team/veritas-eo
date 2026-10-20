# Veritas: Weather Data Trust Monitor

Detects secret tampering (step bias, gradual drift, replay) in a daily NASA POWER
weather series. Each day is compared with what untouched context predicts, and the
system outputs a trust score in [0, 1] and an action (accept / deweight / verify /
quarantine). Downloaded files are fingerprinted, hash-chained and signed.

Status: scaffold only. Full run instructions will be added in the final packaging step.

## Install

```bash
conda env create -f environment.yml
conda activate eo-trust
# or, with exact versions:
pip install -r requirements.lock
```

## Test

```bash
pytest -q
```

Tests run offline on synthetic data.

## Layout

```text
config/    settings.yaml (all settings), tuned.yaml (written by tuning)
src/       library code (data, trust, attacks, detect, respond, eval)
tests/     pytest tests
data/      cached downloads, clean frames, provenance log
results/   benchmark tables and charts
report/    the Phase 1 report
```
