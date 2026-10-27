# Benchmark summary

Test year 2024; attacks start in 2024-02-01..2024-09-30; pipeline warm-up 365 days; metrics use the test year only.

## Standard scenario

| group | attack | n_scenarios | catch_rate | median_delay_days | attribution_acc | median_abs_start_error_days | fa_per_location_year | precision | recall | f1 | roc_auc | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | none | 5 | - | - | - | - | 1.996 | 0.000 | - | - | - | 0.058 |
| test | step | 100 | 1.000 | 2.500 | 0.820 | 3.000 | 1.515 | 0.972 | 0.920 | 0.946 | 0.986 | 0.042 |
| test | drift | 100 | 1.000 | 21.000 | 0.080 | 18.000 | 1.515 | 0.970 | 0.843 | 0.902 | 0.939 | 0.079 |
| test | replay | 100 | 1.000 | 2.000 | 0.280 | 1.000 | 2.598 | 0.153 | 0.587 | 0.242 | 0.870 | 0.046 |
| test | adaptive_step | 50 | 0.800 | 61.000 | 0.775 | 81.500 | 1.511 | 0.926 | 0.345 | 0.503 | 0.778 | 0.288 |

## Unseen year (test year on train locations)

| group | attack | n_scenarios | catch_rate | median_delay_days | attribution_acc | median_abs_start_error_days | fa_per_location_year | precision | recall | f1 | roc_auc | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unseen_year | none | 10 | - | - | - | - | 1.098 | 0.000 | - | - | - | 0.020 |
| unseen_year | step | 200 | 1.000 | 2.500 | 0.675 | 3.000 | 2.244 | 0.947 | 0.948 | 0.948 | 0.990 | 0.036 |
| unseen_year | drift | 200 | 1.000 | 19.000 | 0.080 | 17.000 | 2.244 | 0.942 | 0.862 | 0.900 | 0.951 | 0.075 |
| unseen_year | replay | 200 | 1.000 | 2.000 | 0.160 | 1.000 | 1.954 | 0.211 | 0.637 | 0.317 | 0.904 | 0.034 |

## Step size sweep (catch rate)

| size | catch_rate |
|---|---|
| -0.200 | 1.000 |
| -0.120 | 1.000 |
| -0.080 | 1.000 |
| -0.050 | 0.920 |
| -0.030 | 0.840 |
| 0.030 | 0.800 |
| 0.050 | 1.000 |
| 0.080 | 1.000 |
| 0.120 | 1.000 |
| 0.160 | 1.000 |
| 0.200 | 1.000 |

## Drift sweep (median delay of caught drifts, days)

| size | duration | delay_days |
|---|---|---|
| 0.1 | 7.0 | 10.0 |
| 0.1 | 30.0 | 33.0 |
| 0.1 | 60.0 | 55.0 |
| 0.1 | 120.0 | 87.0 |
| 0.1 | 180.0 | 88.0 |
| 0.1 | 7.0 | 6.0 |
| 0.1 | 30.0 | 16.0 |
| 0.1 | 60.0 | 32.0 |
| 0.1 | 120.0 | 49.0 |
| 0.1 | 180.0 | 62.0 |
| 0.1 | 7.0 | 4.0 |
| 0.1 | 30.0 | 14.0 |
| 0.1 | 60.0 | 23.0 |
| 0.1 | 120.0 | 32.0 |
| 0.1 | 180.0 | 46.0 |
| 0.2 | 7.0 | 3.0 |
| 0.2 | 30.0 | 8.0 |
| 0.2 | 60.0 | 15.0 |
| 0.2 | 120.0 | 22.0 |
| 0.2 | 180.0 | 33.0 |

## Replay sweep (catch rate)

| duration | noise | catch_rate |
|---|---|---|
| 3.000 | 0.000 | 0.640 |
| 3.000 | 0.005 | 0.640 |
| 3.000 | 0.010 | 0.600 |
| 7.000 | 0.000 | 1.000 |
| 7.000 | 0.005 | 0.920 |
| 7.000 | 0.010 | 0.920 |
| 14.000 | 0.000 | 1.000 |
| 14.000 | 0.005 | 1.000 |
| 14.000 | 0.010 | 1.000 |

Measured step detection limit (>= 50% caught): 3%; adaptive attacker uses 0.9 x limit = 2.7% (row adaptive_step).

## Cost

- median 0.330 ms per location-day (1965 runs)
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

Provenance log head hash: `98f0db7d5b3dd0af6ebaba21b10e36be42150b051194bd8178be5c4c55eae6b8`

## Notes

- alarm = action verify or quarantine; caught = an alarm within [start, end + 30 days]
- false alarms = alarm episodes starting on days that are neither attacked nor in the 30-day grace period after the attack; per location-year of such days
- one clean run per location (the pipeline is deterministic)
- per-day precision/recall/F1/AUC/Brier pool the test-year days of each row's runs (grace days excluded)
- policy thresholds: {'accept': 0.8, 'deweight': 0.5, 'verify': 0.2, 'hold_days': 3}
