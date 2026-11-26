# Benchmark summary

Test year 2024; attacks start in 2024-02-01..2024-09-30; pipeline warm-up 365 days; metrics use the test year only.

## Standard scenario

| group | attack | n_scenarios | catch_rate | median_delay_days | attribution_acc | attribution_acc_30d | median_abs_start_error_days | fa_per_location_year | precision | recall | f1 | roc_auc | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | none | 5 | - | - | - | - | - | 0.200 | 0.000 | - | - | - | 0.002 |
| test | step | 100 | 1.000 | 2.000 | 0.940 | 0.840 | 1.000 | 0.000 | 1.000 | 0.826 | 0.905 | 0.984 | 0.085 |
| test | drift | 100 | 1.000 | 10.000 | 0.160 | 0.350 | 7.000 | 0.000 | 1.000 | 0.821 | 0.902 | 0.959 | 0.090 |
| test | replay | 100 | 1.000 | 5.000 | 0.480 | 1.000 | 0.000 | 0.244 | 0.584 | 0.426 | 0.493 | 0.880 | 0.014 |
| test | adaptive_step | 50 | 1.000 | 4.500 | 1.000 | 0.920 | 5.000 | 0.000 | 1.000 | 0.650 | 0.788 | 0.877 | 0.176 |

## Unseen year (test year on train locations)

| group | attack | n_scenarios | catch_rate | median_delay_days | attribution_acc | attribution_acc_30d | median_abs_start_error_days | fa_per_location_year | precision | recall | f1 | roc_auc | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unseen_year | none | 18 | - | - | - | - | - | 0.055 | 0.000 | - | - | - | 0.001 |
| unseen_year | step | 360 | 1.000 | 3.000 | 0.903 | 0.881 | 2.000 | 0.000 | 1.000 | 0.907 | 0.952 | 0.993 | 0.051 |
| unseen_year | drift | 360 | 1.000 | 15.000 | 0.142 | 0.503 | 10.000 | 0.000 | 1.000 | 0.847 | 0.917 | 0.965 | 0.082 |
| unseen_year | replay | 360 | 1.000 | 6.000 | 0.561 | 1.000 | 0.000 | 0.003 | 0.772 | 0.352 | 0.483 | 0.902 | 0.014 |

## Step size sweep (catch rate)

| step_size | catch_rate |
|---|---|
| -20% | 100% |
| -12% | 92% |
| -8% | 84% |
| -5% | 60% |
| -3% | 40% |
| +3% | 100% |
| +5% | 100% |
| +8% | 100% |
| +12% | 100% |
| +16% | 100% |
| +20% | 100% |

## Drift sweep (median delay of caught drifts, days)

| final_size | ramp_days | median_delay_days |
|---|---|---|
| 5% | 7 | 5 |
| 5% | 30 | 13 |
| 5% | 60 | 16 |
| 5% | 120 | 19 |
| 5% | 180 | 37 |
| 10% | 7 | 3 |
| 10% | 30 | 6 |
| 10% | 60 | 13 |
| 10% | 120 | 15 |
| 10% | 180 | 24 |
| 15% | 7 | 3 |
| 15% | 30 | 5 |
| 15% | 60 | 8 |
| 15% | 120 | 11 |
| 15% | 180 | 21 |
| 25% | 7 | 2 |
| 25% | 30 | 4 |
| 25% | 60 | 6 |
| 25% | 120 | 8 |
| 25% | 180 | 18 |

## Replay sweep (catch rate)

| replay_days | noise | catch_rate |
|---|---|---|
| 3 | 0.0% | 28% |
| 3 | 0.5% | 28% |
| 3 | 1.0% | 28% |
| 7 | 0.0% | 100% |
| 7 | 0.5% | 96% |
| 7 | 1.0% | 96% |
| 14 | 0.0% | 100% |
| 14 | 0.5% | 100% |
| 14 | 1.0% | 100% |

Measured step detection limit (>= 50% caught): 3%; adaptive attacker uses 0.9 x limit = 2.7% (row adaptive_step).

## Cost

- median 0.285 ms per location-day (2453 runs)
- peak traced memory 4.7 MB (clean runs, tracemalloc)

## System

- os: Windows 10
- os_version: 10.0.22621
- machine: AMD64
- processor: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
- cores_physical: 8
- cores_logical: 12
- ram_gb: 31.7
- python: 3.11.18
- numpy: 2.4.6
- pandas: 3.1.0

Provenance log head hash: `f5368050c207f622701fdeffd32e683589580b87995c95f1dbb71a8c482bc4e0`

## Notes

- alarm = action verify or quarantine; caught = an alarm within [start, end + 30 days]
- false alarms = alarm episodes starting on days that are neither attacked nor in the 30-day grace period after the attack; per location-year of such days
- one clean run per location (the pipeline is deterministic)
- per-day precision/recall/F1/AUC/Brier pool the test-year days of each row's runs (grace days excluded)
- policy thresholds: {'accept': 0.8, 'deweight': 0.5, 'verify': 0.2, 'hold_days': 3, 'sticky': True}
