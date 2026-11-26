# White-box red team

The attacker can run VERITAS itself on every candidate attack (full knowledge).

## Safe-attack envelope: largest constant bias that stays hidden (days)

| site | direction | 30_days_bias | 90_days_bias | 180_days_bias |
|---|---|---|---|---|
| abidjan | down | 8.2% | 7.4% | 7.4% |
| abidjan | up | 2.3% | 2.3% | 2.3% |
| alice_springs | down | 1.8% | 1.8% | 1.8% |
| alice_springs | up | 0.2% | 0.0% | 0.0% |
| kairouan | down | 10.4% | 6.8% | 5.1% |
| kairouan | up | 0.2% | 0.0% | 0.0% |
| lyon | down | 24.8% | 17.4% | 8.6% |
| lyon | up | 7.6% | 3.1% | 0.2% |
| pergamino | down | 8.6% | 7.6% | 7.6% |
| pergamino | up | 0.0% | 0.0% | 0.0% |

## Greedy adaptive attacker (bias chosen week by week)

| site | direction | days | bias_mean | bias_max | bias_final | caught_day |
|---|---|---|---|---|---|---|
| kairouan | up | 180 | 1.9% | 21.5% | 1.2% | never |
| kairouan | down | 180 | 2.8% | 20.3% | 0.0% | 56 |
| alice_springs | up | 180 | 1.5% | 19.9% | 0.0% | never |
| alice_springs | down | 180 | 0.2% | 5.5% | 0.0% | 7 |
| lyon | up | 180 | 12.7% | 49.6% | 0.4% | never |
| lyon | down | 180 | 13.6% | 48.8% | 0.0% | 140 |
| abidjan | up | 180 | 3.6% | 13.3% | 0.0% | 91 |
| abidjan | down | 180 | 2.0% | 18.0% | 0.0% | 42 |
| pergamino | up | 180 | 5.6% | 48.0% | 0.0% | never |
| pergamino | down | 180 | 9.2% | 20.3% | 9.8% | never |

Median across sites: a constant upward lie must stay below 0.0% to remain hidden for 180 days (range 0.0% - 2.3%).
