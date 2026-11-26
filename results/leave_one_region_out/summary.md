# Leave-one-regime-out (train sites, 2024)

| regime | attack | n | catch_rate | median_delay_days | fa_per_location_year |
|---|---|---|---|---|---|
| cold | none | 2 | - | - | 0.000 |
| cold | step | 20 | 1.000 | 4.000 | 0.000 |
| cold | drift | 20 | 1.000 | 17.000 | 0.000 |
| cold | replay | 20 | 1.000 | 5.500 | 0.000 |
| highland | none | 3 | - | - | 0.000 |
| highland | step | 30 | 1.000 | 8.000 | 0.000 |
| highland | drift | 30 | 1.000 | 36.000 | 0.000 |
| highland | replay | 30 | 1.000 | 6.000 | 0.000 |
| monsoon_tropical | none | 3 | - | - | 0.000 |
| monsoon_tropical | step | 30 | 1.000 | 26.000 | 0.000 |
| monsoon_tropical | drift | 30 | 1.000 | 49.000 | 0.000 |
| monsoon_tropical | replay | 30 | 1.000 | 6.000 | 0.000 |
| arid | none | 4 | - | - | 0.000 |
| arid | step | 40 | 1.000 | 1.000 | 0.000 |
| arid | drift | 40 | 1.000 | 6.000 | 0.000 |
| arid | replay | 40 | 1.000 | 5.000 | 0.000 |
| temperate | none | 6 | - | - | 0.000 |
| temperate | step | 60 | 1.000 | 3.000 | 0.000 |
| temperate | drift | 60 | 1.000 | 11.000 | 0.000 |
| temperate | replay | 60 | 1.000 | 5.500 | 0.000 |

Only the context model is retrained without the held-out regime; the detectors use the frozen tuned thresholds.
