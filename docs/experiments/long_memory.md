\# Experiment: long-memory gap detector for tiny attacks (Phase 2)



\*\*Goal.\*\* Catch small, patient biases (the +2.7% adaptive attack) by averaging the calibrated gap over months.



\*\*Method.\*\* Training locations, training years only (`python -m scripts.measure\_long\_memory`).

(1) Noise of the L-day mean gap on clean data; (2) how much of an injected +2.7% step stays in the gap.



| Window | Std of mean gap | p99.5 abs |

|---|---|---|

| 30 d | 0.0215 | 0.0731 |

| 90 d | 0.0142 | 0.0573 |

| 180 d | 0.0106 | 0.0487 |

| 365 d | 0.0080 | 0.0266 |



Injected +2.7% step (log shift 0.0266): \*\*100% still visible\*\* after 14, 30, 90, 180 and 300 days.



\*\*Findings.\*\*

1\. The per-site calibration does not absorb slow attacks: a patient attacker is never "learned" as normal.

2\. A long-window detector is not viable: the clean gap's slow wander (site/model mismatch, strongest at black\_forest) exceeds the attack size for every window up to 180 days (signal-to-noise 0.36-0.55), so thresholds low enough to catch +2.7% would raise false alarms.

3\. The adaptive attack is already detected in 100% of runs (median 4.5 days); its weak metric is per-day recall (0.63), which points to the response policy, not detection.



\*\*Decision.\*\* No long-memory detector. Next: a "sticky" response policy that keeps a quarantined site under verification while the evidence of tampering persists.

