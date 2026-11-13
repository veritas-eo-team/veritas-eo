"""Compare VERITAS with baselines on the standard scenario (test locations).

python -m src.eval.run_baselines [--n-seeds 20] [--budget 0.6] [--quick]

Same scenarios, seeds and metrics as run_benchmark. Baseline thresholds are tuned on
the training locations to the false-alarm budget. Writes results/baselines/comparison.csv
and comparison.md (results/benchmark_table.csv is not touched).
"""
from __future__ import annotations

import argparse
import time
from typing import Sequence

import pandas as pd

from src.common import PATHS, get_logger, load_config
from src.data.align import list_locations, load
from src.detect.context_model import ContextModel
from src.eval import metrics
from src.eval.baselines import METHODS, TUNED_KINDS, Baselines
from src.eval.run_benchmark import md_table, run_scenarios
from src.eval.scenarios import DEFAULT_ATTACKS, eval_settings, make_scenarios

log = get_logger(__name__)
SHOW = ["method", "attack", "n_scenarios", "catch_rate", "median_delay_days", "fa_per_location_year", "f1", "roc_auc"]


def main(argv: Sequence[str] | None = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=eval_settings(cfg)["n_seeds"])
    ap.add_argument("--budget", type=float, default=0.6, help="false-alarm budget per location-year")
    ap.add_argument("--quick", action="store_true", help="2 seeds")
    a = ap.parse_args(argv)

    model = ContextModel.load()
    test = {i: load(i, cfg) for i in list_locations("test", cfg)}
    train = {i: load(i, cfg) for i in list_locations("train", cfg)}
    n_seeds = min(a.n_seeds, 2) if a.quick else a.n_seeds
    t0 = time.perf_counter()

    base = Baselines(cfg, train)
    print("Baseline thresholds (tuned on training locations):")
    for kind in TUNED_KINDS:
        thr, rate = base.tune(kind, train, a.budget)
        print(f"  {METHODS[kind]:<38} threshold {thr:.4f}  clean alarms {rate:.2f} per location-year")

    scns = make_scenarios("test", DEFAULT_ATTACKS, n_seeds, cfg, sweep=False,
                          locations=list(test), quick=a.quick, group="test")
    tables = []
    for kind, label in METHODS.items():
        recs, days = run_scenarios(scns, test, model, cfg, pipeline=base.pipeline(kind))
        table = metrics.summarize(recs, days)
        table.insert(0, "method", label)
        tables.append(table)
        log.info("done: %s", label)
    out = pd.concat(tables, ignore_index=True)

    folder = PATHS["results"] / "baselines"
    folder.mkdir(parents=True, exist_ok=True)
    out.to_csv(folder / "comparison.csv", index=False)
    show = [c for c in SHOW if c in out.columns]
    (folder / "comparison.md").write_text(
        "# VERITAS vs baselines (test locations, standard scenario)\n\n"
        f"Baseline thresholds tuned on training locations to <= {a.budget} clean alarm episodes "
        "per location-year. Baselines do not label the attack type.\n\n" + md_table(out[show]) + "\n",
        encoding="utf-8")
    print()
    print(out[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\ndone in {time.perf_counter() - t0:.0f} s; wrote results/baselines/comparison.csv and comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())