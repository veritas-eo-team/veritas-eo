# White-box red team

The attacker can run VERITAS itself on every candidate attack (full knowledge).

## Safe-attack envelope: largest constant bias that stays hidden (days)

| site | direction | 30_days_bias | 90_days_bias | 180_days_bias |
|---|---|---|---|---|
| chiang_mai | down | 4.7% | 3.5% | 3.5% |
| chiang_mai | up | 0.0% | 0.0% | 0.0% |
| guangzhou | down | 14.8% | 12.3% | 5.9% |
| guangzhou | up | 5.3% | 5.3% | 5.3% |
| helsinki | down | 14.3% | 10.5% | 8.4% |
| helsinki | up | 3.1% | 0.8% | 0.4% |
| la_paz | down | 12.9% | 6.8% | 4.1% |
| la_paz | up | 9.8% | 1.4% | 0.6% |
| lhasa | down | 18.2% | 12.9% | 12.5% |
| lhasa | up | 0.8% | 0.8% | 0.8% |
| niamey | down | 4.9% | 3.7% | 3.7% |
| niamey | up | 1.4% | 0.8% | 0.8% |
| riyadh | down | 3.9% | 2.0% | 1.4% |
| riyadh | up | 0.0% | 0.0% | 0.0% |
| winnipeg | down | 17.6% | 9.0% | 7.4% |
| winnipeg | up | 1.2% | 1.2% | 0.2% |

## Greedy adaptive attacker (bias chosen week by week)

| site | direction | days | bias_mean | bias_max | bias_final | caught_day |
|---|---|---|---|---|---|---|
| helsinki | up | 180 | 7.0% | 49.6% | 0.0% | 105 |
| winnipeg | up | 180 | 7.1% | 32.0% | 6.6% | never |
| lhasa | up | 180 | 10.0% | 21.5% | 7.4% | never |
| la_paz | up | 180 | 5.5% | 28.9% | 5.9% | never |
| chiang_mai | up | 180 | 6.5% | 18.4% | 12.9% | never |
| niamey | up | 180 | 3.1% | 14.5% | 3.5% | never |
| guangzhou | up | 180 | 15.8% | 49.6% | 13.7% | never |
| riyadh | up | 180 | 0.6% | 8.6% | 0.8% | never |

Median across sites: a constant upward lie must stay below 0.5% to remain hidden for 180 days (range 0.0% - 5.3%).
