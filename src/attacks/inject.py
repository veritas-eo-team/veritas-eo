"""Attack injection (contract C2).

inject(series, attack, ...) returns (attacked copy, labels). Only the given
series changes, NaN stays NaN, and every random choice comes from `seed`.

Attacks
  step    from `start` to the end of the series, every value x (1 + size)   [size 0.12]
  drift   multiplier rises linearly 1 -> 1 + size over `duration` days, then held
                                                              [size 0.15, duration 60]
  replay  `duration` earlier days copied over the window       [duration 7]
          params source: "lastyear" (365 days earlier, default), "recent" (the window
          just before), "random" (any earlier window); noise = relative Gaussian noise
          on the copied values (0.01 means x (1 + N(0, 0.01)))
  spike   x (1 + size) for `duration` days                     [size 0.5, duration 1]
  stuck   the start-day value held for `duration` days         [duration 7]

Smart attacks: adaptive_step() stays just under a known detection limit;
slow_ramp() is a drift over a very long duration.

CLI: python -m src.attacks.inject --demo  ->  results/attack_examples.png
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.common import PATHS, get_logger

log = get_logger(__name__)

ATTACKS: tuple[str, ...] = ("none", "step", "drift", "replay", "spike", "stuck")
DEFAULT_SIZE: dict[str, float] = {"step": 0.12, "drift": 0.15, "spike": 0.5}
DEFAULT_DURATION: dict[str, int] = {"drift": 60, "replay": 7, "spike": 1, "stuck": 7}
REPLAY_SOURCES: tuple[str, ...] = ("lastyear", "recent", "random")
LASTYEAR_LAG = 365


# ------------------------------------------------------------------ helpers

def _check_series(series: pd.Series) -> None:
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    idx = series.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")
    if len(idx) < 2:
        raise ValueError("series must have at least 2 days")
    if not bool((idx[1:] - idx[:-1] == pd.Timedelta(days=1)).all()):
        raise ValueError("series must have a continuous daily index")


def _jsonable(value: Any) -> Any:
    """Make label params JSON-friendly (dates -> 'YYYY-MM-DD', numpy -> Python)."""
    if isinstance(value, (pd.Timestamp, dt.datetime)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _position(index: pd.DatetimeIndex, when: Any) -> int:
    ts = pd.Timestamp(when)
    if ts not in index:
        raise ValueError(f"{ts.date()} is outside the series")
    return int(index.get_loc(ts))


def _resolve_size(attack: str, size: float | None) -> float | None:
    if attack not in DEFAULT_SIZE:
        return None
    value = DEFAULT_SIZE[attack] if size is None else float(size)
    if value <= -1.0:
        raise ValueError("size must be greater than -1 (the multiplier must stay positive)")
    return value


def _resolve_duration(attack: str, duration: int | None) -> int | None:
    if attack == "step":
        return None  # a step lasts until the end of the series
    value = DEFAULT_DURATION[attack] if duration is None else int(duration)
    if value < 1:
        raise ValueError("duration must be at least 1 day")
    return value


def _earliest_start(attack: str, duration: int | None, source: str | None) -> int:
    """Earliest start position that leaves the history the attack needs."""
    if attack == "replay":
        return LASTYEAR_LAG if source == "lastyear" else int(duration)
    return 0


def _random_start(
    index: pd.DatetimeIndex,
    rng: np.random.Generator,
    earliest: int,
    span: int,
    window_start: Any,
    window_end: Any,
) -> int:
    """Random start inside [window_start, window_end] that leaves room for the attack."""
    lo, hi = earliest, len(index) - span
    if window_start is not None:
        lo = max(lo, int(index.searchsorted(pd.Timestamp(window_start))))
    if window_end is not None:
        hi = min(hi, int(index.searchsorted(pd.Timestamp(window_end), side="right")) - 1)
    if lo > hi:
        raise ValueError("no room for the attack in the requested window")
    return int(rng.integers(lo, hi + 1))


def _held_value(x: np.ndarray, start: int) -> float:
    """Value a stuck sensor repeats: the start-day value, else the last valid one before."""
    if not np.isnan(x[start]):
        return float(x[start])
    previous = x[:start][~np.isnan(x[:start])]
    if previous.size == 0:
        raise ValueError("no valid value at or before the start to hold")
    return float(previous[-1])


def _replay_source(start: int, duration: int, source: str, rng: np.random.Generator) -> int:
    if source == "lastyear":
        return start - LASTYEAR_LAG
    if source == "recent":
        return start - duration
    return int(rng.integers(0, start - duration + 1))  # any window ending before the start


# ------------------------------------------------------------------ public API

def inject(
    series: pd.Series,
    attack: str,
    *,
    size: float | None = None,
    start: Any = None,
    duration: int | None = None,
    noise: float = 0.0,
    seed: int = 0,
    **params: Any,
) -> tuple[pd.Series, dict[str, Any]]:
    """Inject one attack into a copy of `series` (contract C2).

    params understood: source (replay), window_start / window_end (random start range),
    decimals (round attacked values, e.g. 2 like NASA POWER). Any other params are
    stored in labels["params"] as tags. noise applies to replay only.

    labels = {type, start, end, size, params} with Timestamps (end = last attacked day);
    replay labels also have source_start.
    """
    if attack not in ATTACKS:
        raise ValueError(f"attack must be one of {ATTACKS}")
    _check_series(series)
    if noise < 0:
        raise ValueError("noise must be >= 0")

    if attack == "none":
        clean = series.copy()
        return clean, {"type": "none", "start": None, "end": None, "size": None,
                       "params": _jsonable({"seed": seed, **params})}

    index = series.index
    n = len(index)
    x = series.to_numpy(dtype=float, copy=True)
    rng = np.random.default_rng(seed)

    source = None
    if attack == "replay":
        source = params.pop("source", "lastyear")
        if source not in REPLAY_SOURCES:
            raise ValueError(f"source must be one of {REPLAY_SOURCES}")
    window_start = params.pop("window_start", None)
    window_end = params.pop("window_end", None)
    decimals = params.pop("decimals", None)

    size_value = _resolve_size(attack, size)
    dur = _resolve_duration(attack, duration)
    span = 1 if attack == "step" else int(dur)
    earliest = _earliest_start(attack, dur, source)

    if start is None:
        s = _random_start(index, rng, earliest, span, window_start, window_end)
        start_mode = "random"
    else:
        s = _position(index, start)
        start_mode = "given"
        if s < earliest:
            raise ValueError(f"{attack} needs {earliest} days of history before the start")
        if s + span > n:
            raise ValueError("the attack does not fit before the end of the series")

    out = x.copy()
    extra: dict[str, Any] = {}
    src = None
    if attack == "step":
        out[s:] = x[s:] * (1.0 + size_value)
        end = n - 1
    elif attack == "drift":
        k = np.arange(n - s, dtype=float)
        out[s:] = x[s:] * (1.0 + size_value * np.minimum((k + 1.0) / dur, 1.0))
        end = n - 1
    elif attack == "spike":
        out[s:s + dur] = x[s:s + dur] * (1.0 + size_value)
        end = s + dur - 1
    elif attack == "stuck":
        held = _held_value(x, s)
        window = x[s:s + dur]
        out[s:s + dur] = np.where(np.isnan(window), np.nan, held)
        end = s + dur - 1
        extra["held_value"] = held
    else:  # replay
        src = _replay_source(s, dur, source, rng)
        copied = x[src:src + dur].copy()
        if noise > 0:
            copied = copied * (1.0 + rng.normal(0.0, noise, dur))
        window = x[s:s + dur]
        # NaN stays NaN; a missing source value leaves the original in place
        out[s:s + dur] = np.where(np.isnan(window), np.nan, np.where(np.isnan(copied), window, copied))
        end = s + dur - 1

    if decimals is not None:
        out[s:end + 1] = np.round(out[s:end + 1], int(decimals))

    attacked = pd.Series(out, index=index, name=series.name)
    attacked.attrs = dict(series.attrs)

    label_params: dict[str, Any] = {"duration": dur, "noise": float(noise), "seed": int(seed),
                                    "start_mode": start_mode}
    if source is not None:
        label_params["source"] = source
    if window_start is not None:
        label_params["window_start"] = window_start
    if window_end is not None:
        label_params["window_end"] = window_end
    if decimals is not None:
        label_params["decimals"] = int(decimals)
    label_params.update(extra)
    label_params.update(params)

    labels: dict[str, Any] = {
        "type": attack,
        "start": index[s],
        "end": index[end],
        "size": size_value,
        "params": _jsonable(label_params),
    }
    if src is not None:
        labels["source_start"] = index[src]
        labels["params"]["source_start"] = _jsonable(index[src])
    return attacked, labels


def attack_mask(labels: Mapping[str, Any], index: pd.Index) -> pd.Series:
    """Boolean Series on `index`: True on attacked days (start..end inclusive)."""
    idx = pd.DatetimeIndex(index)
    if labels.get("type", "none") == "none" or labels.get("start") is None:
        return pd.Series(False, index=idx, name="attacked")
    start, end = pd.Timestamp(labels["start"]), pd.Timestamp(labels["end"])
    return pd.Series((idx >= start) & (idx <= end), index=idx, name="attacked")


def adaptive_step(
    series: pd.Series,
    limit: float,
    margin: float = 0.9,
    *,
    start: Any = None,
    seed: int = 0,
    **params: Any,
) -> tuple[pd.Series, dict[str, Any]]:
    """A step sized margin x limit: just under a detector's known detection limit."""
    if limit == 0:
        raise ValueError("limit must be non-zero")
    if not 0 < margin <= 1:
        raise ValueError("margin must be in (0, 1]")
    return inject(series, "step", size=margin * float(limit), start=start, seed=seed,
                  adaptive=True, limit=float(limit), margin=float(margin), **params)


def slow_ramp(
    series: pd.Series,
    size: float,
    duration_days: int,
    *,
    start: Any = None,
    seed: int = 0,
    **params: Any,
) -> tuple[pd.Series, dict[str, Any]]:
    """A drift over a very long duration (labelled type "drift")."""
    return inject(series, "drift", size=size, duration=int(duration_days), start=start, seed=seed,
                  slow=True, **params)


# ------------------------------------------------------------------ demo

def _demo_series(location: str) -> tuple[pd.Series, str]:
    """Real clean target if available, else synthetic data."""
    try:
        from src.data.align import clean_path, load
        if clean_path(location).exists():
            df = load(location)
            return df["target"], f"{location} ({df.attrs.get('units') or 'target'})"
    except Exception as exc:  # demo only: fall back quietly
        log.warning("could not load %s (%s); using synthetic data", location, exc)
    from src.synthetic import make_synthetic_frame
    return make_synthetic_frame(seed=0)["target"], "synthetic data"


def demo(out_path: Path | str | None = None, location: str = "tunis") -> Path:
    """Plot clean vs attacked for each attack type; returns the PNG path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series, title = _demo_series(location)
    index = series.index
    candidate = pd.Timestamp(f"{index[-1].year}-04-01")
    start = candidate if candidate in index and index.get_loc(candidate) >= LASTYEAR_LAG else index[len(index) // 2]
    cases = [
        ("step", {}, "step +12%"),
        ("drift", {}, "drift to +15% over 60 days"),
        ("replay", {"source": "recent"}, "replay of 7 days"),
        ("spike", {}, "spike +50% for 1 day"),
        ("stuck", {}, "stuck for 7 days"),
    ]
    fig, axes = plt.subplots(len(cases), 1, figsize=(10, 2.4 * len(cases)))
    for ax, (attack, kwargs, label) in zip(axes, cases):
        attacked, labels = inject(series, attack, start=start, seed=1, **kwargs)
        lo = labels["start"] - pd.Timedelta(days=30)
        open_ended = attack in ("step", "drift")
        hi = labels["start"] + pd.Timedelta(days=120) if open_ended else labels["end"] + pd.Timedelta(days=30)
        view = slice(lo, hi)
        ax.plot(series[view], lw=1.0, label="clean")
        ax.plot(attacked[view], lw=1.0, label="attacked")
        ax.axvspan(labels["start"], min(labels["end"], hi), color="red", alpha=0.12, label="attacked days")
        if "source_start" in labels:
            src_end = labels["source_start"] + pd.Timedelta(days=labels["params"]["duration"] - 1)
            ax.axvspan(labels["source_start"], src_end, color="grey", alpha=0.25, label="copied from")
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Standard attacks on {title}", fontsize=11)
    fig.tight_layout()
    path = Path(out_path) if out_path is not None else PATHS["results"] / "attack_examples.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m src.attacks.inject --demo [--location tunis]"""
    parser = argparse.ArgumentParser(description="Attack injection tools.")
    parser.add_argument("--demo", action="store_true", help="plot each attack to results/attack_examples.png")
    parser.add_argument("--location", default="tunis", help="location for the demo (falls back to synthetic)")
    args = parser.parse_args(argv)
    if not args.demo:
        parser.print_help()
        return 0
    path = demo(location=args.location)
    print(f"saved {path.name} to the results folder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())