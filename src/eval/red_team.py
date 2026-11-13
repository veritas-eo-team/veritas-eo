"""White-box red team (Phase 2, Step 12).

The attacker has full access to VERITAS (model, settings, code) and uses it as an
oracle: every candidate attack is run through the real pipeline before it is chosen.

1. Safe-attack envelope: per site, the largest constant bias (up and down) that stays
   undetected (no verify/quarantine) for each horizon, found by bisection.
2. Greedy adaptive attacker: week by week, the largest bias that keeps the whole run
   alarm-free up to the end of that week; earlier weeks stay fixed.

Each site uses the first start date whose clean run has no alarm in the window, so
natural false alarms are never counted as "detecting" the attacker.

python -m src.eval.red_team [--quick] [--down]
Writes results/red_team/{envelope.csv, greedy.csv, envelope.png, greedy_paths.png, summary.md}.
"""
from __future__ import annotations

import argparse
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.common import PATHS, get_logger, load_config
from src.pipeline import ALARM_ACTIONS, run_pipeline

log = get_logger(__name__)

HORIZONS: tuple[int, ...] = (30, 90, 180)
MAX_SIZE = 0.5
BISECT_STEPS = 8
BLOCK_DAYS = 7
GREEDY_STEPS = 7
START_CANDIDATES: tuple[str, ...] = ("03-01", "04-01", "05-01", "02-01")


class Oracle:
    """VERITAS on one site with a multiplier applied from `start` on (white-box access)."""

    def __init__(self, frame: pd.DataFrame, model: Any, cfg: Mapping[str, Any], start: Any,
                 horizon: int, warmup: int = 365) -> None:
        self.model, self.cfg = model, cfg
        self.start = pd.Timestamp(start)
        begin = max(frame.index[0], self.start - pd.Timedelta(days=warmup))
        end = self.start + pd.Timedelta(days=int(horizon) - 1)
        if end > frame.index[-1]:
            raise ValueError("the horizon runs past the end of the data")
        self.df = frame.loc[begin:end].copy()
        self.clean = self.df["target"].to_numpy(dtype=float)
        self.pos = int(self.df.index.get_loc(self.start))
        self.horizon = int(horizon)
        self.runs = 0

    def first_alarm(self, mult: Any) -> int | None:
        """Days from the start to the first verify/quarantine, or None if none in the window.

        mult: scalar or array (one value per day from the start), applied to the clean target.
        """
        t = self.clean.copy()
        t[self.pos:] = t[self.pos:] * np.asarray(mult, dtype=float)
        df = self.df.copy()
        df["target"] = t
        res = run_pipeline(df, self.model, self.cfg)
        self.runs += 1
        hits = np.flatnonzero(res["action"].isin(ALARM_ACTIONS).to_numpy()[self.pos:])
        return int(hits[0]) if hits.size else None


def clean_oracle(frame: pd.DataFrame, model: Any, cfg: Mapping[str, Any], horizon: int,
                 candidates: Sequence[str] = START_CANDIDATES) -> Oracle | None:
    """First start in the test year whose clean run has no alarm in the window."""
    ty = int(cfg["test_year"])
    for mmdd in candidates:
        try:
            oracle = Oracle(frame, model, cfg, f"{ty}-{mmdd}", horizon)
        except ValueError:
            continue
        if oracle.first_alarm(1.0) is None:
            return oracle
    return None


def max_undetected(oracle: Oracle, horizon: int, sign: int = 1, cache: dict | None = None,
                   steps: int = BISECT_STEPS, max_size: float = MAX_SIZE) -> float:
    """Largest constant bias (fraction) with no alarm in the first `horizon` days."""
    cache = cache if cache is not None else {}

    def hidden(size: float) -> bool:
        key = round(size, 6)
        if key not in cache:
            cache[key] = oracle.first_alarm(1.0 + sign * size)
        fa = cache[key]
        return fa is None or fa >= horizon

    lo, hi = 0.0, float(max_size)
    if hidden(hi):
        return hi
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        if hidden(mid):
            lo = mid
        else:
            hi = mid
    return lo


def greedy_attack(oracle: Oracle, horizon: int | None = None, sign: int = 1, block: int = BLOCK_DAYS,
                  steps: int = GREEDY_STEPS, max_size: float = MAX_SIZE) -> tuple[np.ndarray, int | None]:
    """Week-by-week largest hidden bias. Returns (bias per day, day caught or None)."""
    n = int(horizon or oracle.horizon)
    mult = np.ones(len(oracle.clean) - oracle.pos)
    caught = None
    for b0 in range(0, n, block):
        b1 = min(n, b0 + block)

        def hidden(size: float) -> bool:
            m = mult.copy()
            m[b0:b1] = 1.0 + sign * size
            fa = oracle.first_alarm(m)
            return fa is None or fa >= b1

        if not hidden(0.0):          # earlier weeks already give the attacker away
            caught = b0
            break
        lo, hi = 0.0, float(max_size)
        for _ in range(steps):
            mid = (lo + hi) / 2.0
            if hidden(mid):
                lo = mid
            else:
                hi = mid
        mult[b0:b1] = 1.0 + sign * lo
    return (mult[:n] - 1.0), caught


# ------------------------------------------------------------------ reporting

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_envelope(env: pd.DataFrame, path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (site, sign), d in env.groupby(["site", "direction"]):
        style = "-" if sign == "up" else "--"
        ax.plot(d["horizon_days"], d["max_hidden_bias"] * 100, style, marker="o", label=f"{site} ({sign})")
    ax.set_xlabel("days the attacker must stay hidden")
    ax.set_ylabel("largest hidden constant bias (%)")
    ax.set_title("Safe-attack envelope (white-box attacker)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_greedy(paths: Mapping[str, np.ndarray], path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for site, bias in paths.items():
        ax.plot(np.arange(len(bias)), bias * 100, lw=1.3, label=site)
    ax.set_xlabel("days since the attack started")
    ax.set_ylabel("bias chosen by the attacker (%)")
    ax.set_title("Greedy white-box attacker: largest bias that stays hidden")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.1%}" if isinstance(v, float) and abs(v) <= 1 and c.endswith(("bias", "_mean", "_max", "_final")) else str(v)
                                        for v, c in zip(row, cols)) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ main

def main(argv: Sequence[str] | None = None) -> int:
    from src.data.align import list_locations, load
    from src.detect.context_model import ContextModel

    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--locations", nargs="*", help="default: the test locations")
    ap.add_argument("--down", action="store_true", help="also run the greedy attacker downward")
    ap.add_argument("--quick", action="store_true", help="two sites, horizons 30 and 90")
    a = ap.parse_args(argv)

    model = ContextModel.load()
    sites = a.locations or list_locations("test", cfg)
    horizons = (30, 90) if a.quick else HORIZONS
    if a.quick:
        sites = sites[:2]
    longest = max(horizons)
    t0 = time.perf_counter()

    env_rows, greedy_rows, paths = [], [], {}
    for site in sites:
        frame = load(site, cfg)
        oracle = clean_oracle(frame, model, cfg, longest)
        if oracle is None:
            print(f"{site}: every candidate start has a natural false alarm in the window; skipped")
            continue
        for sign, name in ((1, "up"), (-1, "down")):
            cache: dict = {}
            for h in horizons:
                env_rows.append({"site": site, "start": oracle.start.date(), "direction": name,
                                 "horizon_days": h, "max_hidden_bias": max_undetected(oracle, h, sign, cache)})
        for sign, name in ((1, "up"), (-1, "down")) if a.down else ((1, "up"),):
            bias, caught = greedy_attack(oracle, longest, sign)
            key = site if name == "up" else f"{site} (down)"
            paths[key] = bias
            greedy_rows.append({"site": site, "direction": name, "days": longest,
                                "bias_mean": float(np.mean(np.abs(bias))),
                                "bias_max": float(np.max(np.abs(bias))),
                                "bias_final": float(abs(bias[-1])),
                                "caught_day": "never" if caught is None else caught})
        print(f"{site}: done ({oracle.runs} detector runs, start {oracle.start.date()})")

    env = pd.DataFrame(env_rows)
    greedy = pd.DataFrame(greedy_rows)
    out = PATHS["results"] / "red_team"
    out.mkdir(parents=True, exist_ok=True)
    env.to_csv(out / "envelope.csv", index=False)
    greedy.to_csv(out / "greedy.csv", index=False)
    if not env.empty:
        plot_envelope(env, out / "envelope.png")
    if paths:
        plot_greedy(paths, out / "greedy_paths.png")

    pivot = env.pivot_table(index=["site", "direction"], columns="horizon_days", values="max_hidden_bias").reset_index()
    pivot.columns = [str(c) if isinstance(c, int) else c for c in pivot.columns]
    lines = ["# White-box red team", "",
             "The attacker can run VERITAS itself on every candidate attack (full knowledge).", "",
             "## Safe-attack envelope: largest constant bias that stays hidden (days)", "",
             md(pivot.rename(columns={str(h): f"{h}_days_bias" for h in horizons})), "",
             "## Greedy adaptive attacker (bias chosen week by week)", "", md(greedy), ""]
    up = env[(env["direction"] == "up") & (env["horizon_days"] == longest)]["max_hidden_bias"]
    if len(up):
        lines += [f"Median across sites: a constant upward lie must stay below {up.median():.1%} "
                  f"to remain hidden for {longest} days (range {up.min():.1%} - {up.max():.1%}).", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print()
    print(pivot.to_string(index=False, float_format=lambda v: f"{v:.1%}"))
    print()
    print(greedy.to_string(index=False, float_format=lambda v: f"{v:.1%}"))
    print(f"\ndone in {time.perf_counter() - t0:.0f} s; wrote results/red_team/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())