"""Benchmark scenarios.

Each scenario is a JSON-friendly dict: which location, which attack, which seed, and the
window where the (seeded, random) attack start must fall: eval.attack_window inside
test_year. The pipeline later runs on [start - warmup_days, end of test_year] and the
metrics use the test year only.

Families:
  standard  the official attacks: step +12%, drift to +15% over 60 days, 7-day replay,
            plus one clean ("none") run per location (the pipeline is deterministic)
  sweep_*   step sizes +/-3..20%, drifts 5-25% over 7-180 days, replays 3-14 days with
            noise 0 / 0.5% / 1% (fewer seeds: max(1, n_seeds // 4))
  adaptive  adaptive_step at 0.9 of the measured step limit (built after the step sweep)
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.common import load_config

STANDARD: dict[str, dict[str, Any]] = {
    "none": {},
    "step": {"size": 0.12},
    "drift": {"size": 0.15, "duration": 60},
    "replay": {"duration": 7},
}
DEFAULT_ATTACKS: tuple[str, ...] = ("none", "step", "drift", "replay")
SWEEP_STEP_SIZES = (-0.20, -0.12, -0.08, -0.05, -0.03, 0.03, 0.05, 0.08, 0.12, 0.16, 0.20)
SWEEP_DRIFT_SIZES = (0.05, 0.10, 0.15, 0.25)
SWEEP_DRIFT_DURATIONS = (7, 30, 60, 120, 180)
SWEEP_REPLAY_DURATIONS = (3, 7, 14)
SWEEP_REPLAY_NOISE = (0.0, 0.005, 0.01)
QUICK_SWEEP: dict[str, tuple] = {
    "step": (-0.12, 0.05, 0.12),
    "drift": ((0.15, 30),),
    "replay": ((7, 0.01),),
}
ADAPTIVE_MARGIN = 0.9


def eval_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """test_year, attack window (as dates) and warm-up days from config."""
    ev = dict(cfg.get("eval") or {})
    window = ev.get("attack_window") or ["02-01", "09-30"]
    ty = int(cfg["test_year"])
    return {
        "test_year": ty,
        "window_start": f"{ty}-{str(window[0])}",
        "window_end": f"{ty}-{str(window[1])}",
        "warmup_days": int(ev.get("warmup_days") or 365),
        "n_seeds": int(ev.get("n_seeds") or 20),
    }


def _scenario(settings: Mapping[str, Any], group: str, family: str, location: str, attack: str,
              seed: int, size: float | None = None, duration: int | None = None,
              noise: float = 0.0, source: str | None = None,
              adaptive_limit: float | None = None) -> dict[str, Any]:
    sid = f"{group}/{family}/{location}/{attack}/s{size}/d{duration}/n{noise}/seed{seed}"
    return {
        "id": sid, "group": group, "family": family, "location": location, "attack": attack,
        "seed": int(seed), "size": size, "duration": duration, "noise": float(noise),
        "source": source, "adaptive_limit": adaptive_limit,
        "test_year": settings["test_year"], "window_start": settings["window_start"],
        "window_end": settings["window_end"], "warmup_days": settings["warmup_days"],
    }


def make_scenarios(
    split: str = "test",
    attacks: Sequence[str] = DEFAULT_ATTACKS,
    n_seeds: int = 20,
    cfg: Mapping[str, Any] | None = None,
    sweep: bool = False,
    *,
    locations: Sequence[str] | None = None,
    quick: bool = False,
    group: str | None = None,
) -> list[dict[str, Any]]:
    """Standard scenarios (and the sweep if asked) for every location of `split`."""
    cfg = cfg if cfg is not None else load_config()
    settings = eval_settings(cfg)
    if locations is None:
        from src.data.align import list_locations
        locations = list_locations(split, cfg)
    group = group or split
    out: list[dict[str, Any]] = []
    for loc in locations:
        for attack in attacks:
            if attack == "none":
                out.append(_scenario(settings, group, "standard", loc, "none", 0))
                continue
            spec = STANDARD.get(attack, {})
            for seed in range(n_seeds):
                out.append(_scenario(settings, group, "standard", loc, attack, seed,
                                     size=spec.get("size"), duration=spec.get("duration")))
        if sweep:
            out.extend(_sweep(settings, group, loc, n_seeds, quick))
    return out


def _sweep(settings: Mapping[str, Any], group: str, loc: str, n_seeds: int, quick: bool) -> list[dict[str, Any]]:
    seeds = range(1 if quick else max(1, n_seeds // 4))
    steps = QUICK_SWEEP["step"] if quick else SWEEP_STEP_SIZES
    drifts = QUICK_SWEEP["drift"] if quick else [(s, d) for s in SWEEP_DRIFT_SIZES for d in SWEEP_DRIFT_DURATIONS]
    replays = QUICK_SWEEP["replay"] if quick else [(d, n) for d in SWEEP_REPLAY_DURATIONS for n in SWEEP_REPLAY_NOISE]
    out = []
    for seed in seeds:
        out += [_scenario(settings, group, "sweep_step", loc, "step", seed, size=s) for s in steps]
        out += [_scenario(settings, group, "sweep_drift", loc, "drift", seed, size=s, duration=d) for s, d in drifts]
        out += [_scenario(settings, group, "sweep_replay", loc, "replay", seed, duration=d, noise=n)
                for d, n in replays]
    return out


def adaptive_scenarios(limit: float, locations: Sequence[str], n_seeds: int,
                       cfg: Mapping[str, Any] | None = None, group: str = "test") -> list[dict[str, Any]]:
    """adaptive_step scenarios sized ADAPTIVE_MARGIN x the measured detection limit."""
    cfg = cfg if cfg is not None else load_config()
    settings = eval_settings(cfg)
    return [_scenario(settings, group, "adaptive", loc, "step", seed,
                      size=round(ADAPTIVE_MARGIN * limit, 4), adaptive_limit=float(limit))
            for loc in locations for seed in range(n_seeds)]