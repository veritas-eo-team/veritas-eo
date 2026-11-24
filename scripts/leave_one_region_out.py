"""Leave-one-regime-out evaluation (Phase 2).

For each regime the context model is retrained without that regime's sites and the
frozen detectors (config/tuned.yaml) are evaluated on the held-out sites in 2024.
Only the context model changes; thresholds stay as tuned on all train sites.

    python -m scripts.leave_one_region_out [--n-seeds 10]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common import PATHS, get_logger, load_config
from src.data.align import list_locations, load
from src.detect.context_model import ContextModel
from src.eval.run_benchmark import md_table, run_scenarios
from src.eval.scenarios import DEFAULT_ATTACKS, make_scenarios
from src.pipeline import run_pipeline

log = get_logger(__name__)

REGIMES: dict[str, tuple[str, ...]] = {
    "cold": ("helsinki", "winnipeg"),
    "highland": ("lhasa", "la_paz", "nairobi"),
    "monsoon_tropical": ("chiang_mai", "guangzhou", "manaus"),
    "arid": ("tozeur", "phoenix", "riyadh", "niamey"),
    "temperate": ("tunis", "seville", "paris", "black_forest", "ames", "ludhiana"),
}


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    """One row per (regime, attack): catch, median delay and clean false alarms."""
    rows = []
    for regime, g in records.groupby("regime", sort=False):
        clean = g[g["attack"] == "none"]
        fa = float(clean["fa_episodes"].mean()) if len(clean) else float("nan")
        for attack, h in g.groupby("attack", sort=False):
            attacked = attack != "none"
            rows.append({
                "regime": regime,
                "attack": attack,
                "n": len(h),
                "catch_rate": float(h["caught"].mean()) if attacked else float("nan"),
                "median_delay_days": float(h["delay_days"].median()) if attacked else float("nan"),
                "fa_per_location_year": fa,
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--out", default=str(Path(PATHS["results"]) / "leave_one_region_out"))
    args = ap.parse_args(argv)

    cfg = load_config()
    train = list_locations("train", cfg)
    assigned = {loc for held in REGIMES.values() for loc in held}
    unknown = assigned - set(train)
    if unknown:
        raise SystemExit(f"not train locations: {sorted(unknown)}; update REGIMES")
    missing = set(train) - assigned
    if missing:
        log.warning("train locations not in any regime: %s", sorted(missing))

    frames = {loc: load(loc, cfg) for loc in train}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_records = []
    for regime, held in REGIMES.items():
        held = [loc for loc in held if loc in frames]
        fit = [frames[loc] for loc in train if loc not in held]
        model = ContextModel(cfg=cfg).fit(fit)
        scns = make_scenarios("train", DEFAULT_ATTACKS, args.n_seeds, cfg,
                              locations=held, group=f"loro_{regime}")
        recs, _ = run_scenarios(scns, {loc: frames[loc] for loc in held}, model, cfg,
                                pipeline=run_pipeline)
        recs["regime"] = regime
        all_records.append(recs)
        log.info("regime %s: %d scenarios, held out %s", regime, len(recs), held)

    records = pd.concat(all_records, ignore_index=True)
    records.to_csv(out / "per_scenario.csv", index=False)
    table = summarize(records)
    table.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(
        "# Leave-one-regime-out (train sites, 2024)\n\n"
        + md_table(table)
        + "\n\nOnly the context model is retrained without the held-out regime; the detectors "
          "use the frozen tuned thresholds.\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
