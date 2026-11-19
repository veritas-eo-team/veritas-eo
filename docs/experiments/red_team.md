\# White-box red team (Phase 2, Step 12)



\*\*Attacker.\*\* Full knowledge of VERITAS (model, settings, code); every candidate attack is run

through the real pipeline before it is chosen. `python -m src.eval.red\_team --down`.

Test locations, test year 2024, starts chosen so the clean run has no alarm in the window.



\## 1. Safe-attack envelope: largest constant bias hidden for 180 days



| Site | Upward | Downward |

|---|---|---|

| Kairouan | 0.0% | 3.5% |

| Alice Springs | 0.0% | 0.6% |

| Pergamino | 0.0% | 4.5% |

| Lyon | 0.2% | 4.7% |

| Abidjan | 2.3% | 4.3% |

| Median | 0.0% | 4.3% |



Shorter horizons allow more at cloudy sites (Lyon: 16.2% downward for 30 days, 4.7% for 180).



\## 2. Greedy adaptive attacker (bias re-chosen every week, 180 days)



| Site | Direction | Mean bias | Max weekly bias | Caught |

|---|---|---|---|---|

| Kairouan | up / down | 1.5% / 1.9% | 11.7% / 11.7% | never / day 56 |

| Alice Springs | up / down | 0.2% / 0.4% | 1.2% / 4.7% | never / day 21 |

| Lyon | up / down | 2.0% / 12.1% | 22.7% / 41.8% | day 28 / never |

| Abidjan | up / down | 2.5% / 0.9% | 12.5% / 9.8% | day 77 / day 28 |

| Pergamino | up / down | 3.7% / 7.4% | 23.8% / 15.6% | never / never |



\## Findings

1\. Sustained upward lies are close to impossible: the clear-sky ceiling plus the detectors leave

&#x20;  0-2.3% over six months.

2\. Downward lies are the attacker's space: about 4% constant over six months (0.6% at Alice Springs).

3\. Even a full-knowledge attacker was caught in 5 of 10 runs (days 21-77): it checked that no alarm

&#x20;  had fired yet, but the slower detectors kept accumulating evidence it could not foresee.

4\. Weak spot: downward lies at cloudy or variable sites (Lyon mean 12.1%, Pergamino 7.4%, never

&#x20;  caught). A downward lie looks like extra cloud and has no physical floor; Steps 4 and 7 showed

&#x20;  the available witnesses cannot separate the two.



\*\*Caveat.\*\* In this benchmark the sensor is NASA POWER, so clean values never exceed clear-sky and

the ceiling is exact. A real pyranometer (2-5% calibration tolerance) needs a looser clear-sky

tolerance, which would widen the upward envelope to roughly that size.

