\# VERITAS vs standard baselines (Phase 2, Step 10)



\*\*Setup.\*\* `python -m src.eval.run\_baselines --budget 1.0`. Same 305 scenarios as the benchmark

(5 unseen test locations, test year 2024, 20 seeds per attack). Baseline thresholds tuned on the

training locations only, to at most 1.0 clean alarm episode per location-year (VERITAS's clean

rate on the test locations is 0.998).



| Method | Step caught | Drift caught | Replay caught | Step delay (median) | Clean false alarms (test) |

|---|---|---|---|---|---|

| \*\*VERITAS (full)\*\* | \*\*100%\*\* | \*\*100%\*\* | \*\*100%\*\* | \*\*2 days\*\* | 1.0 |

| CUSUM on climatology | 51% | 48% | 7% | 98 days | 1.0 |

| Z-score on climatology (7-day mean) | 44% | 48% | 19% | 45 days | 1.8 |

| Isolation Forest (same inputs as VERITAS) | 0% | 5% | 7% | - | 0.2 |

| VERITAS without neighbours (ablation) | alarms almost continuously: 4.5-6.4 false alarms per location-year | | | | |



At a stricter budget (0.6) the baselines catch 0-44%.



\*\*Findings.\*\*

1\. At equal false alarms, standard change detectors catch about half the attacks and take 1.5-3

&#x20;  months; VERITAS catches all of them, 20-50 times faster.

2\. A generic anomaly detector fails even with the same inputs: a 12% lie looks like an ordinary

&#x20;  day unless the expected value comes from independent witnesses and is tested statistically.

3\. The neighbour witnesses are essential: without them false alarms rise about 10-fold.

