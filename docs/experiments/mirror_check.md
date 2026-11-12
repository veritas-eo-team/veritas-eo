\# Experiment: mirror clear-day check for downward bias (Phase 2)



\*\*Goal.\*\* Catch small downward biases (-3% to -5%), the weak spot of the Phase 1 physics check, which only flags values above the clear-sky limit.



\*\*Version 1: own clear-day clearness vs own history.\*\* On Sentinel-2 clear days, compare clearness (target / clear-sky) with the site's own past clear days.

Result: false alarms rose from 0.40 to 2.55 per location-year on the test sites. A clear Sentinel-2 snapshot at \~10:30 does not mean a clear day; afternoon clouds lower the daily value.



\*\*Version 2: neighbour-relative ratio on clear days.\*\* A day is clear when all four neighbours are near clear-sky (the target is never used to decide). On clear days, compare target / mean(neighbours) with the site's own reference ratio; average over the last N clear days.

Tuned on the 10 training sites, training years only, with a budget of 0.05 clean alarm episodes per location-year:



| Reference | Ratio noise (std, by site) | Smallest safe tolerance |

|---|---|---|

| Rolling (60-365 days ago) | 2.7% - 5.7% | 7.5% (N = 15) |

| Seasonal (same dates, previous 3 years) | 2.6% - 5.3% | 7.5% (N = 15) |



\*\*Conclusion.\*\* Even on clear days, neighbouring NASA POWER cells disagree by 3-5% from day to day, and the disagreement is neither random enough to average away nor seasonal. The check could only flag downward shifts of about 8% or more, which the existing step detector already catches (92% of -8% steps). The check is kept in the code, switched off (`consistency.mirror: false`), and covered by tests, so the experiment can be reproduced with `python -m scripts.tune\_mirror` and `--ref-mode seasonal`.

