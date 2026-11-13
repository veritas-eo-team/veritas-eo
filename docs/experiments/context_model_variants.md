\# Experiment: context-model variants for cloudy sites (Phase 2)



\*\*Goal.\*\* Reduce gap noise at cloudy sites (Lyon, Abidjan), where small attacks are detected late

(+2.7% step: Lyon day 53, Abidjan day 209; sunny sites day 1-9).



\*\*Method.\*\* `python -m scripts.compare\_context\_models`. Calibrated gap std (annual cycle removed, as in

calibrate\_site). Training sites out-of-fold by location; test sites in 2024 after calibration on 2023.



| Variant | Train sites (mean) | Test sites (mean) | Lyon | Abidjan |

|---|---|---|---|---|

| Ridge (current) | 0.0829 | 0.0904 | 0.1584 | 0.0860 |

| Ridge + neighbour extras (min, max, clearness, range) | -0.2% | -0.6% | -0.6% | -0.2% |

| Gradient boosting + neighbour extras | +1.7% | +2.2% | +2.8% | -3.4% |



\*\*Findings.\*\*

1\. The current model already uses what the inputs can tell; the remaining noise is local weather

&#x20;  that neighbouring cells 1 degree away do not see.

2\. The step CUSUM accumulates only shifts above k \~ 1.1 noise units, so its smallest reliably

&#x20;  detected step is about 1.1 x site noise: \~6-7% at sunny sites, \~9.5% at Abidjan and Pergamino,

&#x20;  \~17.5% at Lyon. At sunny sites the clear-sky physics check catches far smaller biases.

3\. Adding NASA POWER's cloud amount for the site cell was rejected: in this benchmark the sensor

&#x20;  values are NASA POWER radiation, which is derived from the same cloud retrievals, so the gain

&#x20;  would be leakage, not a real independent witness.



\*\*Decision.\*\* Keep the current context model. Report a per-site minimum detectable bias instead of

a single headline, and state the cloudy-site limit openly.

