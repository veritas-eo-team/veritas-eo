# VERITAS vs baselines (test locations, standard scenario)

Baseline thresholds tuned on training locations to <= 1.0 clean alarm episodes per location-year. Baselines do not label the attack type.

| method | attack | n_scenarios | catch_rate | median_delay_days | fa_per_location_year | f1 | roc_auc |
|---|---|---|---|---|---|---|---|
| VERITAS (full) | none | 5 | - | - | 0.998 | - | - |
| VERITAS (full) | step | 100 | 1.000 | 2.000 | 0.403 | 0.932 | 0.986 |
| VERITAS (full) | drift | 100 | 1.000 | 10.000 | 0.403 | 0.921 | 0.955 |
| VERITAS (full) | replay | 100 | 1.000 | 4.500 | 0.711 | 0.323 | 0.860 |
| VERITAS without neighbours | none | 5 | - | - | 6.387 | - | - |
| VERITAS without neighbours | step | 100 | 1.000 | 0.000 | 4.469 | 0.661 | 0.617 |
| VERITAS without neighbours | drift | 100 | 1.000 | 0.000 | 4.469 | 0.661 | 0.623 |
| VERITAS without neighbours | replay | 100 | 1.000 | 0.000 | 5.395 | 0.047 | 0.538 |
| CUSUM on climatology | none | 5 | - | - | 0.998 | - | - |
| CUSUM on climatology | step | 100 | 0.510 | 98.000 | 0.000 | 0.289 | 0.763 |
| CUSUM on climatology | drift | 100 | 0.480 | 95.000 | 0.000 | 0.321 | 0.778 |
| CUSUM on climatology | replay | 100 | 0.070 | 30.000 | 0.278 | - | 0.421 |
| Z-score on climatology (7-day mean) | none | 5 | - | - | 1.796 | - | - |
| Z-score on climatology (7-day mean) | step | 100 | 0.440 | 44.500 | 0.805 | 0.030 | 0.950 |
| Z-score on climatology (7-day mean) | drift | 100 | 0.480 | 63.000 | 0.805 | 0.044 | 0.951 |
| Z-score on climatology (7-day mean) | replay | 100 | 0.190 | 15.000 | 1.510 | 0.012 | 0.702 |
| Isolation Forest (same inputs) | none | 5 | - | - | 0.200 | - | - |
| Isolation Forest (same inputs) | step | 100 | 0.000 | - | 0.345 | - | 0.804 |
| Isolation Forest (same inputs) | drift | 100 | 0.050 | 72.000 | 0.345 | 0.001 | 0.795 |
| Isolation Forest (same inputs) | replay | 100 | 0.070 | 4.000 | 0.222 | 0.053 | 0.684 |
