# Benchmark summary

Test year 2024; attacks start in 2024-02-01..2024-09-30; pipeline warm-up 365 days; metrics use the test year only.

## Standard scenario

| group | attack | n_scenarios | catch_rate | median_delay_days | attribution_acc | attribution_acc_30d | median_abs_start_error_days | fa_per_location_year | precision | recall | f1 | roc_auc | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | none | 5 | - | - | - | - | - | 0.998 | 0.000 | - | - | - | 0.033 |
| test | step | 100 | 1.000 | 2.000 | 0.870 | 0.810 | 1.000 | 0.403 | 0.988 | 0.874 | 0.928 | 0.986 | 0.060 |
| test | drift | 100 | 1.000 | 10.000 | 0.160 | 0.350 | 6.000 | 0.403 | 0.988 | 0.858 | 0.918 | 0.955 | 0.069 |
| test | replay | 100 | 1.000 | 4.500 | 0.430 | 1.000 | 0.000 | 0.711 | 0.257 | 0.461 | 0.330 | 0.860 | 0.027 |
| test | adaptive_step | 50 | 1.000 | 4.500 | 0.960 | 0.920 | 5.000 | 0.405 | 0.983 | 0.633 | 0.770 | 0.853 | 0.178 |

## Unseen year (test year on train locations)

| group | attack | n_scenarios | catch_rate | median_delay_days | attribution_acc | attribution_acc_30d | median_abs_start_error_days | fa_per_location_year | precision | recall | f1 | roc_auc | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unseen_year | none | 10 | - | - | - | - | - | 0.599 | 0.000 | - | - | - | 0.006 |
| unseen_year | step | 200 | 1.000 | 3.000 | 0.895 | 0.865 | 2.000 | 0.537 | 0.987 | 0.941 | 0.964 | 0.991 | 0.036 |
| unseen_year | drift | 200 | 1.000 | 13.000 | 0.105 | 0.550 | 6.500 | 0.537 | 0.987 | 0.884 | 0.933 | 0.969 | 0.061 |
| unseen_year | replay | 200 | 1.000 | 4.000 | 0.350 | 1.000 | 0.000 | 0.583 | 0.360 | 0.463 | 0.405 | 0.900 | 0.019 |

## Step size sweep (catch rate)

| step_size | catch_rate |
|---|---|
| -20% | 100% |
| -12% | 100% |
| -8% | 92% |
| -5% | 88% |
| -3% | 80% |
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
| 3 | 0.0% | 36% |
| 3 | 0.5% | 36% |
| 3 | 1.0% | 36% |
| 7 | 0.0% | 100% |
| 7 | 0.5% | 96% |
| 7 | 1.0% | 96% |
| 14 | 0.0% | 100% |
| 14 | 0.5% | 100% |
| 14 | 1.0% | 100% |

Measured step detection limit (>= 50% caught): 3%; adaptive attacker uses 0.9 x limit = 2.7% (row adaptive_step).

## Cost

- median 0.153 ms per location-day (1965 runs)
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

Provenance log head hash: `038ee7c91d72c20521f0b32035a0cf426e98a79486253f8daa9c01d8370ec7af`

## Notes

- alarm = action verify or quarantine; caught = an alarm within [start, end + 30 days]
- false alarms = alarm episodes starting on days that are neither attacked nor in the 30-day grace period after the attack; per location-year of such days
- one clean run per location (the pipeline is deterministic)
- per-day precision/recall/F1/AUC/Brier pool the test-year days of each row's runs (grace days excluded)
- policy thresholds: {'accept': 0.8, 'deweight': 0.5, 'verify': 0.2, 'hold_days': 3}
