# VERITAS vs baselines (test locations, standard scenario)

Baseline thresholds tuned on training locations to <= 0.6 clean alarm episodes per location-year. Baselines do not label the attack type.

| method | attack | n_scenarios | catch_rate | median_delay_days | fa_per_location_year | f1 | roc_auc |
|---|---|---|---|---|---|---|---|
| VERITAS (full) | none | 5 | - | - | 0.200 | - | - |
| VERITAS (full) | step | 100 | 1.000 | 2.000 | 0.000 | 0.905 | 0.984 |
| VERITAS (full) | drift | 100 | 1.000 | 10.000 | 0.000 | 0.902 | 0.959 |
| VERITAS (full) | replay | 100 | 1.000 | 5.000 | 0.244 | 0.493 | 0.880 |
| VERITAS without neighbours | none | 5 | - | - | 6.986 | - | - |
| VERITAS without neighbours | step | 100 | 1.000 | 0.000 | 4.411 | 0.667 | 0.713 |
| VERITAS without neighbours | drift | 100 | 1.000 | 0.000 | 4.411 | 0.672 | 0.718 |
| VERITAS without neighbours | replay | 100 | 1.000 | 0.000 | 5.395 | 0.053 | 0.593 |
| CUSUM on climatology | none | 5 | - | - | 0.000 | - | - |
| CUSUM on climatology | step | 100 | 0.100 | 58.000 | 0.000 | 0.047 | 0.769 |
| CUSUM on climatology | drift | 100 | 0.380 | 134.500 | 0.000 | 0.092 | 0.784 |
| CUSUM on climatology | replay | 100 | 0.000 | - | 0.000 | - | 0.421 |
| Z-score on climatology (7-day mean) | none | 5 | - | - | 0.399 | - | - |
| Z-score on climatology (7-day mean) | step | 100 | 0.160 | 27.500 | 0.268 | 0.013 | 0.950 |
| Z-score on climatology (7-day mean) | drift | 100 | 0.130 | 38.000 | 0.268 | 0.010 | 0.951 |
| Z-score on climatology (7-day mean) | replay | 100 | 0.040 | 8.000 | 0.178 | 0.005 | 0.702 |
| Isolation Forest (same inputs) | none | 5 | - | - | 0.000 | - | - |
| Isolation Forest (same inputs) | step | 100 | 0.000 | - | 0.000 | - | 0.753 |
| Isolation Forest (same inputs) | drift | 100 | 0.000 | - | 0.000 | - | 0.747 |
| Isolation Forest (same inputs) | replay | 100 | 0.080 | 5.000 | 0.000 | 0.045 | 0.707 |
