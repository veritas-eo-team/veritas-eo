"""Deterministic synthetic frames matching contract C1, for offline tests.

The target behaves like daily all-sky solar radiation: a seasonal clear-sky
curve times a clearness index. Clearness is the logistic of a site climate
term plus AR(1) weather noise, so log(target) = log(clear-sky) + a smooth,
autocorrelated weather term, and the target stays below clear-sky. Neighbour
cells share the weather component and add their own noise and a small fixed
cell bias. Sentinel-2-style cloud and NDVI appear every REVISIT_DAYS days.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

TARGET_VAR = "ALLSKY_SFC_SW_DWN"
TARGET_UNITS = "kWh/m^2/day"
CONTEXT_VARS: tuple[str, ...] = (
    "CLRSKY_SFC_SW_DWN", "T2M", "RH2M", "PRECTOTCORR", "GWETTOP",
)
MAX_NEIGHBORS = 4
REVISIT_DAYS = 5
CLEAR_CLOUD_MAX = 0.3

_YEAR = 365.25


def _season(doy: np.ndarray, peak_doy: float) -> np.ndarray:
    """Cosine seasonal cycle equal to 1 on `peak_doy`."""
    return np.cos(2.0 * np.pi * (doy - peak_doy) / _YEAR)


def _ar1(rng: np.random.Generator, n: int, phi: float, sd: float) -> np.ndarray:
    """Stationary AR(1) series with marginal standard deviation `sd`."""
    shocks = rng.normal(0.0, sd * np.sqrt(1.0 - phi**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sd)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + shocks[t]
    return x


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_synthetic_frame(
    location: str = "synthetic",
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    seed: int = 0,
    n_neighbors: int = 4,
    split: str = "train",
    *,
    missing_frac: float = 0.0,
) -> pd.DataFrame:
    """Build a C1-shaped frame from a seeded generator.

    The random stream depends on (seed, location), and every component is always
    drawn, so the target is identical whatever `n_neighbors` or `missing_frac` is.
    `missing_frac` sets a random share of target days to NaN.
    """
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not 0 <= n_neighbors <= MAX_NEIGHBORS:
        raise ValueError(f"n_neighbors must be between 0 and {MAX_NEIGHBORS}")
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    if not 0.0 <= missing_frac < 1.0:
        raise ValueError("missing_frac must be in [0, 1)")

    index = pd.date_range(start, end, freq="D", name="date")
    n = len(index)
    if n < 2:
        raise ValueError("the date range must cover at least 2 days")

    rng = np.random.default_rng([int(seed), zlib.crc32(location.encode("utf-8"))])
    doy = index.dayofyear.to_numpy(dtype=float)

    # Site climate (varies by location and seed).
    clr_base = rng.uniform(6.0, 7.5)
    clr_amp = rng.uniform(1.0, 3.0)
    clear_bias = rng.uniform(0.6, 1.8)
    temp_mean = rng.uniform(8.0, 22.0)
    temp_amp = rng.uniform(5.0, 12.0)

    clr_true = clr_base + clr_amp * _season(doy, 172.0)
    logit_mean = clear_bias + 0.4 * _season(doy, 172.0)

    # Weather: shared component plus own noise for the site and each neighbour.
    shared = _ar1(rng, n, phi=0.6, sd=0.8)
    own = np.stack([_ar1(rng, n, phi=0.3, sd=0.35) for _ in range(1 + MAX_NEIGHBORS)])
    cell_bias = rng.normal(0.0, 0.03, MAX_NEIGHBORS)

    clearness = _sigmoid(logit_mean + shared + own[0])
    target = clr_true * clearness
    neighbours = [
        clr_true * (1.0 + cell_bias[i]) * _sigmoid(logit_mean + shared + own[i + 1])
        for i in range(MAX_NEIGHBORS)
    ]

    # Other NASA variables at the site.
    clrsky = clr_true * (1.0 + rng.normal(0.0, 0.01, n))
    cloudiness = 1.0 - clearness
    rain_prob = np.clip(1.5 * cloudiness - 0.3, 0.0, 1.0)
    rain = rng.uniform(size=n) < rain_prob
    precip = np.where(rain, rng.exponential(5.0, n) * cloudiness, 0.0)
    gwet = np.empty(n)
    level = 0.4
    for t in range(n):
        level = min(1.0, max(0.05, 0.97 * level + 0.015 * precip[t]))
        gwet[t] = level
    t2m = temp_mean + temp_amp * _season(doy, 200.0) + 2.0 * shared + rng.normal(0.0, 1.5, n)
    rh2m = np.clip(
        65.0 - 10.0 * _season(doy, 200.0) - 6.0 * shared + 15.0 * cloudiness
        + rng.normal(0.0, 4.0, n),
        5.0, 100.0,
    )

    # Sentinel-2: acquisitions every REVISIT_DAYS days; high cloud on dark days.
    offset = int(rng.integers(0, REVISIT_DAYS))
    acquired = (np.arange(n) % REVISIT_DAYS) == offset
    cloud_all = np.round(np.clip(1.2 * cloudiness + rng.normal(0.0, 0.05, n), 0.0, 1.0), 3)
    cloud = np.where(acquired, cloud_all, np.nan)
    clear_acq = acquired & (cloud_all <= CLEAR_CLOUD_MAX)

    ndvi_true = 0.45 + 0.2 * _season(doy, 150.0) + _ar1(rng, n, phi=0.98, sd=0.05)
    ndvi_obs = np.round(np.clip(ndvi_true + rng.normal(0.0, 0.02, n), -0.2, 0.95), 4)
    ndvi = pd.Series(np.where(clear_acq, ndvi_obs, np.nan)).ffill().to_numpy()
    position = np.arange(n, dtype=float)
    last_clear = pd.Series(np.where(clear_acq, position, np.nan)).ffill().to_numpy()
    days_since_clear = position - last_clear

    missing = rng.uniform(size=n) < missing_frac
    target = np.where(missing, np.nan, np.round(target, 2))

    data: dict[str, np.ndarray] = {"target": target}
    for i in range(n_neighbors):
        data[f"nb_{i}"] = np.round(neighbours[i], 2)
    context = {
        "CLRSKY_SFC_SW_DWN": clrsky,
        "T2M": t2m,
        "RH2M": rh2m,
        "PRECTOTCORR": precip,
        "GWETTOP": gwet,
    }
    for var in CONTEXT_VARS:
        data[f"ctx_{var}"] = np.round(context[var], 2)
    data["ndvi"] = ndvi
    data["cloud"] = cloud
    data["days_since_clear"] = days_since_clear

    frame = pd.DataFrame(data, index=index).astype("float64")
    frame.attrs.update(
        location=location,
        lat=40.0,
        lon=0.0,
        climate="synthetic",
        split=split,
        target_var=TARGET_VAR,
        units=TARGET_UNITS,
    )
    return frame
