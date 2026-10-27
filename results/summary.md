# Benchmark summary

Test year 2024; attacks start in 2024-02-01..2024-09-30; pipeline warm-up 365 days; metrics use the test year only.

## Standard scenario

| group | attack | n_scenarios | catch_rate | median_delay_days | attribution_acc | attribution_acc_30d | median_abs_start_error_days | fa_per_location_year | precision | recall | f1 | roc_auc | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | none | 5 | - | - | - | - | - | 0.998 | 0.000 | - | - | - | 0.033 |
| test | step | 100 | 1.000 | 2.000 | 0.980 | 0.860 | 1.000 | 0.403 | 0.988 | 0.874 | 0.928 | 0.986 | 0.060 |
| test | drift | 100 | 1.000 | 10.000 | 0.070 | 0.200 | 6.000 | 0.403 | 0.988 | 0.858 | 0.918 | 0.955 | 0.069 |
| test | replay | 100 | 1.000 | 4.500 | 0.430 | 1.000 | 0.000 | 0.711 | 0.257 | 0.461 | 0.330 | 0.860 | 0.027 |
| test | adaptive_step | 50 | 1.000 | 4.500 | 1.000 | 1.000 | 5.000 | 0.405 | 0.983 | 0.633 | 0.770 | 0.853 | 0.178 |

## Unseen year (test year on train locations)

| group | attack | n_scenarios | catch_rate | median_delay_days | attribution_acc | attribution_acc_30d | median_abs_start_error_days | fa_per_location_year | precision | recall | f1 | roc_auc | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unseen_year | none | 10 | - | - | - | - | - | 0.599 | 0.000 | - | - | - | 0.006 |
| unseen_year | step | 200 | 1.000 | 3.000 | 0.960 | 0.970 | 2.000 | 0.537 | 0.987 | 0.941 | 0.964 | 0.991 | 0.036 |
| unseen_year | drift | 200 | 1.000 | 13.000 | 0.030 | 0.315 | 6.500 | 0.537 | 0.987 | 0.884 | 0.933 | 0.969 | 0.061 |
| unseen_year | replay | 200 | 1.000 | 4.000 | 0.350 | 1.000 | 0.000 | 0.583 | 0.360 | 0.463 | 0.405 | 0.900 | 0.019 |

## Step size sweep (catch rate)

| size | catch_rate |
|---|---|
| -0.200 | 1.000 |
| -0.120 | 1.000 |
| -0.080 | 0.920 |
| -0.050 | 0.880 |
| -0.030 | 0.800 |
| 0.030 | 1.000 |
| 0.050 | 1.000 |
| 0.080 | 1.000 |
| 0.120 | 1.000 |
| 0.160 | 1.000 |
| 0.200 | 1.000 |

## Drift sweep (median delay of caught drifts, days)

| size | duration | delay_days |
|---|---|---|
| 0.1 | 7.0 | 5.0 |
| 0.1 | 30.0 | 13.0 |
| 0.1 | 60.0 | 16.0 |
| 0.1 | 120.0 | 19.0 |
| 0.1 | 180.0 | 37.0 |
| 0.1 | 7.0 | 3.0 |
| 0.1 | 30.0 | 6.0 |
| 0.1 | 60.0 | 13.0 |
| 0.1 | 120.0 | 15.0 |
| 0.1 | 180.0 | 24.0 |
| 0.1 | 7.0 | 3.0 |
| 0.1 | 30.0 | 5.0 |
| 0.1 | 60.0 | 8.0 |
| 0.1 | 120.0 | 11.0 |
| 0.1 | 180.0 | 21.0 |
| 0.2 | 7.0 | 2.0 |
| 0.2 | 30.0 | 4.0 |
| 0.2 | 60.0 | 6.0 |
| 0.2 | 120.0 | 8.0 |
| 0.2 | 180.0 | 18.0 |

## Replay sweep (catch rate)

| duration | noise | catch_rate |
|---|---|---|
| 3.000 | 0.000 | 0.360 |
| 3.000 | 0.005 | 0.360 |
| 3.000 | 0.010 | 0.360 |
| 7.000 | 0.000 | 1.000 |
| 7.000 | 0.005 | 0.960 |
| 7.000 | 0.010 | 0.960 |
| 14.000 | 0.000 | 1.000 |
| 14.000 | 0.005 | 1.000 |
| 14.000 | 0.010 | 1.000 |

Measured step detection limit (>= 50% caught): 3%; adaptive attacker uses 0.9 x limit = 2.7% (row adaptive_step).

## Cost

- median 0.298 ms per location-day (1965 runs)
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
