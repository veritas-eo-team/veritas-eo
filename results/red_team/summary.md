# White-box red team

The attacker can run VERITAS itself on every candidate attack (full knowledge).

## Safe-attack envelope: largest constant bias that stays hidden (days)

| site | direction | 30_days_bias | 90_days_bias | 180_days_bias |
|---|---|---|---|---|
| abidjan | down | 4.7% | 4.3% | 4.3% |
| abidjan | up | 2.3% | 2.3% | 2.3% |
| alice_springs | down | 3.3% | 3.1% | 0.6% |
| alice_springs | up | 0.0% | 0.0% | 0.0% |
| kairouan | down | 6.1% | 3.9% | 3.5% |
| kairouan | up | 0.2% | 0.0% | 0.0% |
| lyon | down | 16.2% | 9.8% | 4.7% |
| lyon | up | 7.6% | 3.1% | 0.2% |
| pergamino | down | 5.3% | 4.5% | 4.5% |
| pergamino | up | 0.0% | 0.0% | 0.0% |

## Greedy adaptive attacker (bias chosen week by week)

| site | direction | days | bias_mean | bias_max | bias_final | caught_day |
|---|---|---|---|---|---|---|
| kairouan | up | 180 | 1.5% | 11.7% | 1.2% | never |
| kairouan | down | 180 | 1.9% | 11.7% | 0.0% | 56 |
| alice_springs | up | 180 | 0.2% | 1.2% | 1.2% | never |
| alice_springs | down | 180 | 0.4% | 4.7% | 0.0% | 21 |
| lyon | up | 180 | 2.0% | 22.7% | 0.0% | 28 |
| lyon | down | 180 | 12.1% | 41.8% | 1.2% | never |
| abidjan | up | 180 | 2.5% | 12.5% | 0.0% | 77 |
| abidjan | down | 180 | 0.9% | 9.8% | 0.0% | 28 |
| pergamino | up | 180 | 3.7% | 23.8% | 0.0% | never |
| pergamino | down | 180 | 7.4% | 15.6% | 9.4% | never |

Median across sites: a constant upward lie must stay below 0.0% to remain hidden for 180 days (range 0.0% - 2.3%).
