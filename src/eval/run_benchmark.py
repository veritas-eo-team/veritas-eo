"""Run the benchmark and write results/.

python -m src.eval.run_benchmark [--split test] [--n-seeds 20] [--quick] [--sweep]

Main group: the locations of --split (test by default). With --split test it also
reports "unseen_year": the test year on the train locations (the context model only
trained on earlier years). Writes benchmark_table.csv, per_scenario.csv, summary.md,
system_info.json and the charts catch_vs_size.png, delay_vs_drift_duration.png,
false_alarms_vs_threshold.png, reliability.png.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.common import PATHS, get_logger, load_config
from src.eval import metrics
from src.eval.scenarios import ADAPTIVE_MARGIN, DEFAULT_ATTACKS, adaptive_scenarios, eval_settings, make_scenarios
from src.pipeline import run_pipeline, timed_run

log = get_logger(__name__)

THRESHOLDS = np.round(np.arange(0.30, 0.96, 0.05), 2)


# ------------------------------------------------------------------ running

def run_scenario(scn: Mapping[str, Any], frame: pd.DataFrame, model: Any, cfg: Mapping[str, Any],
                 measure_memory: bool = False) -> tuple[dict[str, Any], pd.DataFrame]:
    """Inject, run the pipeline on [start - warmup, end of test year], score the test year."""
    from src.attacks.inject import adaptive_step, inject

    ty = int(scn["test_year"])
    series = frame["target"]
    kw: dict[str, Any] = {"seed": scn["seed"], "window_start": scn["window_start"], "window_end": scn["window_end"]}
    if scn["family"] == "adaptive":
        attacked, labels = adaptive_step(series, scn["adaptive_limit"], ADAPTIVE_MARGIN, **kw)
    else:
        if scn["size"] is not None:
            kw["size"] = scn["size"]
        if scn["duration"] is not None:
            kw["duration"] = scn["duration"]
        if scn["attack"] == "replay":
            kw["noise"] = scn["noise"]
            if scn["source"]:
                kw["source"] = scn["source"]
        attacked, labels = inject(series, scn["attack"], **kw)

    anchor = labels["start"] if labels["start"] is not None else pd.Timestamp(f"{ty}-01-01")
    begin = max(frame.index[0], anchor - pd.Timedelta(days=int(scn["warmup_days"])))
    run_df = frame.loc[begin:f"{ty}-12-31"].copy()
    run_df["target"] = attacked.loc[run_df.index]

    if measure_memory:
        result, ms_per_day, peak_mb = timed_run(run_df, model, cfg)
    else:
        t0 = time.perf_counter()
        result = run_pipeline(run_df, model, cfg)
        ms_per_day = (time.perf_counter() - t0) * 1000.0 / max(len(run_df), 1)
        peak_mb = float("nan")

    record = {**scn, "start": labels["start"], "end": labels["end"],
              **metrics.event_metrics(result, labels, ty),
              "ms_per_day": ms_per_day, "peak_mb": peak_mb, "n_days_run": int(len(run_df))}
    r = result.loc[result.index.year == ty]
    y, grace = metrics.attack_windows(r.index, labels)
    perday = pd.DataFrame({"scenario_id": scn["id"], "y": y, "grace": grace,
                           "alarm": metrics.alarm_mask(r), "p": r["p_compromise"].to_numpy()})
    return record, perday


def run_scenarios(scenarios: Sequence[Mapping[str, Any]], frames: Mapping[str, pd.DataFrame], model: Any,
                  cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, days = [], []
    for i, scn in enumerate(scenarios):
        rec, perday = run_scenario(scn, frames[scn["location"]], model, cfg,
                                   measure_memory=scn["attack"] == "none")
        records.append(rec)
        days.append(perday)
        if (i + 1) % 25 == 0 or i + 1 == len(scenarios):
            log.info("ran %d / %d scenarios", i + 1, len(scenarios))
    if not records:
        return pd.DataFrame(), pd.DataFrame(columns=["scenario_id", "y", "grace", "alarm", "p"])
    return pd.DataFrame(records), pd.concat(days, ignore_index=True)


# ------------------------------------------------------------------ analysis

def threshold_tradeoff(records: pd.DataFrame, perday: pd.DataFrame) -> pd.DataFrame:
    """False alarms per location-year and catch rate if the alarm were p >= threshold (no hysteresis)."""
    groups = [(sid, d) for sid, d in perday[perday["scenario_id"].isin(records["id"])].groupby("scenario_id", sort=False)]
    rows = []
    for thr in THRESHOLDS:
        fa_ep = fa_days = 0
        caught: list[bool] = []
        for _, d in groups:
            a = d["p"].to_numpy() >= thr
            y, g = d["y"].to_numpy(), d["grace"].to_numpy()
            clean = ~y & ~g
            fa_ep += int(np.sum(metrics.rising_edges(a) & clean))
            fa_days += int(clean.sum())
            if y.any():
                caught.append(bool(np.any(a & (y | g))))
        rows.append({"threshold": float(thr), "fa_per_location_year": metrics.fa_per_location_year(fa_ep, fa_days),
                     "catch_rate": float(np.mean(caught)) if caught else np.nan})
    return pd.DataFrame(rows)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    import matplotlib.pyplot as plt
    plt.close(fig)


def chart_catch_vs_size(records: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    d = records[(records["attack"] == "step") & records["family"].isin(["standard", "sweep_step"])]
    rates = d.groupby("size")["caught"].apply(lambda s: s.astype(float).mean()).sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rates.index * 100, rates.values, marker="o")
    ax.axvline(12, color="grey", ls=":", lw=1)
    ax.set_xlabel("step size (%)")
    ax.set_ylabel("catch rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Catch rate vs step size")
    _save(fig, path)


def chart_delay_vs_drift(records: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    d = records[(records["attack"] == "drift") & (records["caught"] == True)]  # noqa: E712
    fig, ax = plt.subplots(figsize=(6, 4))
    for size, sub in d.groupby("size"):
        med = sub.groupby("duration")["delay_days"].median().sort_index()
        ax.plot(med.index, med.values, marker="o", label=f"{size:.0%}")
    if d.empty:
        ax.text(0.5, 0.5, "no caught drifts", ha="center", transform=ax.transAxes)
    else:
        ax.legend(title="final size", fontsize=8)
    ax.set_xlabel("drift duration (days)")
    ax.set_ylabel("median delay (days)")
    ax.set_title("Detection delay vs drift duration")
    _save(fig, path)


def chart_threshold(tradeoff: pd.DataFrame, policy_p: float, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tradeoff["threshold"], tradeoff["fa_per_location_year"], marker="o", color="tab:red")
    ax.set_xlabel("alarm threshold on p_compromise")
    ax.set_ylabel("false alarms per location-year", color="tab:red")
    ax2 = ax.twinx()
    ax2.plot(tradeoff["threshold"], tradeoff["catch_rate"], marker="s", color="tab:blue")
    ax2.set_ylabel("catch rate (standard attacks)", color="tab:blue")
    ax2.set_ylim(-0.05, 1.05)
    ax.axvline(policy_p, color="grey", ls=":", lw=1)
    ax.set_title("False alarms and catch rate vs threshold")
    _save(fig, path)


def chart_reliability(table: pd.DataFrame, brier: float, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 5))
    ok = table["n"] > 0
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
    ax.plot(table.loc[ok, "mean_p"], table.loc[ok, "frac_positive"], marker="o")
    ax.set_xlabel("mean predicted p_compromise")
    ax.set_ylabel("fraction of attacked days")
    ax.set_title(f"Reliability (Brier {brier:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, path)


# ------------------------------------------------------------------ reporting

def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    """Plain markdown table without extra dependencies."""
    def fmt(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _provenance_head() -> str | None:
    try:
        from src.trust.verify import verify_log
        return verify_log(check_files=False).head_hash
    except Exception:
        return None


def write_summary(path: Path, table: pd.DataFrame, records: pd.DataFrame, settings: Mapping[str, Any],
                  info: Mapping[str, Any], cfg: Mapping[str, Any], main_group: str,
                  limit: float | None) -> None:
    lines = ["# Benchmark summary", ""]
    lines += [f"Test year {settings['test_year']}; attacks start in {settings['window_start']}..{settings['window_end']}; "
              f"pipeline warm-up {settings['warmup_days']} days; metrics use the test year only.", ""]
    lines += ["## Standard scenario", "", md_table(table[table["group"] == main_group]), ""]
    if (table["group"] == "unseen_year").any():
        lines += ["## Unseen year (test year on train locations)", "",
                  md_table(table[table["group"] == "unseen_year"]), ""]
    sweep = records[records["family"].str.startswith("sweep") & (records["group"] == main_group)]
    if not sweep.empty:
        step = sweep[sweep["family"] == "sweep_step"].groupby("size")["caught"].apply(
            lambda s: s.astype(float).mean()).reset_index()
        step_tab = pd.DataFrame({
            "step_size": [f"{v * 100:+.0f}%" for v in step["size"]],
            "catch_rate": [f"{v:.0%}" for v in step["caught"]],
        })
        lines += ["## Step size sweep (catch rate)", "", md_table(step_tab), ""]
        drift = sweep[(sweep["family"] == "sweep_drift") & (sweep["caught"] == True)]  # noqa: E712
        if not drift.empty:
            tab = drift.groupby(["size", "duration"])["delay_days"].median().reset_index()
            drift_tab = pd.DataFrame({
                "final_size": [f"{v:.0%}" for v in tab["size"]],
                "ramp_days": [f"{int(v)}" for v in tab["duration"]],
                "median_delay_days": [f"{v:g}" for v in tab["delay_days"]],
            })
            lines += ["## Drift sweep (median delay of caught drifts, days)", "", md_table(drift_tab), ""]
        rep = sweep[sweep["family"] == "sweep_replay"].groupby(["duration", "noise"])["caught"].apply(
            lambda s: s.astype(float).mean()).reset_index()
        rep_tab = pd.DataFrame({
            "replay_days": [f"{int(v)}" for v in rep["duration"]],
            "noise": [f"{v:.1%}" for v in rep["noise"]],
            "catch_rate": [f"{v:.0%}" for v in rep["caught"]],
        })
        lines += ["## Replay sweep (catch rate)", "", md_table(rep_tab), ""]
        drift = sweep[(sweep["family"] == "sweep_drift") & (sweep["caught"] == True)]  # noqa: E712
        if not drift.empty:
            tab = drift.groupby(["size", "duration"])["delay_days"].median().reset_index()
            drift_tab = pd.DataFrame({
                "final_size": [f"{v:.0%}" for v in tab["size"]],
                "ramp_days": [f"{int(v)}" for v in tab["duration"]],
                "median_delay_days": [f"{v:g}" for v in tab["delay_days"]],
            })
            lines += ["## Drift sweep (median delay of caught drifts, days)", "", md_table(drift_tab), ""]
        rep = sweep[sweep["family"] == "sweep_replay"].groupby(["duration", "noise"])["caught"].apply(
            lambda s: s.astype(float).mean()).reset_index()
        rep_tab = pd.DataFrame({
            "replay_days": [f"{int(v)}" for v in rep["duration"]],
            "noise": [f"{v:.1%}" for v in rep["noise"]],
            "catch_rate": [f"{v:.0%}" for v in rep["caught"]],
        })
        lines += ["## Replay sweep (catch rate)", "", md_table(rep_tab), ""]
    if limit is not None:
        lines += [f"Measured step detection limit (>= 50% caught): {limit:.0%}; adaptive attacker uses "
                  f"{ADAPTIVE_MARGIN} x limit = {ADAPTIVE_MARGIN * limit:.1%} (row adaptive_step).", ""]
    clean = records[records["attack"] == "none"]
    lines += ["## Cost", "",
              f"- median {records['ms_per_day'].median():.3f} ms per location-day "
              f"({len(records)} runs)",
              f"- peak traced memory {clean['peak_mb'].max():.1f} MB (clean runs, tracemalloc)", ""]
    lines += ["## System", ""] + [f"- {k}: {v}" for k, v in info.items()] + [""]
    head = _provenance_head()
    if head:
        lines += [f"Provenance log head hash: `{head}`", ""]
    pol = dict(cfg.get("policy") or {})
    lines += ["## Notes", "",
              "- alarm = action verify or quarantine; caught = an alarm within [start, end + 30 days]",
              "- false alarms = alarm episodes starting on days that are neither attacked nor in the 30-day "
              "grace period after the attack; per location-year of such days",
              "- one clean run per location (the pipeline is deterministic)",
              "- per-day precision/recall/F1/AUC/Brier pool the test-year days of each row's runs (grace days excluded)",
              f"- policy thresholds: {pol}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ main entry

def run(cfg: Mapping[str, Any], model: Any, test_frames: Mapping[str, pd.DataFrame],
        train_frames: Mapping[str, pd.DataFrame] | None = None, n_seeds: int = 20, sweep: bool = False,
        quick: bool = False, out_dir: Path | str | None = None, split: str = "test",
        attacks: Sequence[str] = DEFAULT_ATTACKS) -> dict[str, Any]:
    """Run all scenarios, write the result files, return {table, records, paths}."""
    out = Path(out_dir) if out_dir is not None else PATHS["results"]
    out.mkdir(parents=True, exist_ok=True)
    settings = eval_settings(cfg)
    groups: list[tuple[str, Mapping[str, pd.DataFrame]]] = [(split, test_frames)]
    if split == "test" and train_frames:
        groups.append(("unseen_year", train_frames))

    all_records, all_days = [], []
    limit = None
    for group, frames in groups:
        is_main = group == split
        scns = make_scenarios(split, attacks, n_seeds, cfg, sweep=sweep and is_main,
                              locations=list(frames), quick=quick, group=group)
        recs, days = run_scenarios(scns, frames, model, cfg)
        all_records.append(recs)
        all_days.append(days)
        if sweep and is_main:
            limit = metrics.measured_limit(recs)
            if limit is not None:
                adapt = adaptive_scenarios(limit, list(frames), 1 if quick else max(1, n_seeds // 2), cfg, group)
                recs2, days2 = run_scenarios(adapt, frames, model, cfg)
                all_records.append(recs2)
                all_days.append(days2)

    records = pd.concat(all_records, ignore_index=True)
    perday = pd.concat(all_days, ignore_index=True)
    table = metrics.summarize(records, perday)
    info = metrics.system_info()

    paths = {name: out / name for name in (
        "benchmark_table.csv", "per_scenario.csv", "summary.md", "system_info.json",
        "catch_vs_size.png", "delay_vs_drift_duration.png", "false_alarms_vs_threshold.png", "reliability.png")}
    table.to_csv(paths["benchmark_table.csv"], index=False)
    records.to_csv(paths["per_scenario.csv"], index=False, date_format="%Y-%m-%d")
    paths["system_info.json"].write_text(json.dumps(info, indent=2), encoding="utf-8")

    main_recs = records[records["group"] == split]
    main_std = main_recs[main_recs["family"] == "standard"]
    main_days = perday[perday["scenario_id"].isin(main_std["id"]) & ~perday["grace"]]
    brier, rel = metrics.calibration(main_days["y"], main_days["p"])
    policy_p = 1.0 - float((cfg.get("policy") or {}).get("deweight", 0.5))
    chart_catch_vs_size(main_recs, paths["catch_vs_size.png"])
    chart_delay_vs_drift(main_recs, paths["delay_vs_drift_duration.png"])
    chart_threshold(threshold_tradeoff(main_std, perday), policy_p, paths["false_alarms_vs_threshold.png"])
    chart_reliability(rel, brier, paths["reliability.png"])
    write_summary(paths["summary.md"], table, records, settings, info, cfg, split, limit)
    return {"table": table, "records": records, "paths": paths, "limit": limit}


def main(argv: Sequence[str] | None = None) -> int:
    from src.data.align import list_locations, load
    from src.detect.context_model import ContextModel

    cfg = load_config()
    parser = argparse.ArgumentParser(description="Run the benchmark and write results/.")
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--n-seeds", type=int, default=eval_settings(cfg)["n_seeds"])
    parser.add_argument("--quick", action="store_true", help="2 seeds and a small sweep")
    parser.add_argument("--sweep", action="store_true", help="also run the size / duration / noise sweep")
    args = parser.parse_args(argv)

    try:
        model = ContextModel.load()
    except FileNotFoundError as exc:
        print(exc)
        return 1
    n_seeds = min(args.n_seeds, 2) if args.quick else args.n_seeds
    test_frames = {i: load(i, cfg) for i in list_locations(args.split, cfg)}
    train_frames = {i: load(i, cfg) for i in list_locations("train", cfg)} if args.split == "test" else None
    t0 = time.perf_counter()
    res = run(cfg, model, test_frames, train_frames, n_seeds=n_seeds, sweep=args.sweep,
              quick=args.quick, split=args.split)
    cols = ["group", "attack", "n_scenarios", "catch_rate", "median_delay_days",
            "attribution_acc", "attribution_acc_30d", "fa_per_location_year", "f1", "roc_auc"]
    print(res["table"][cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"done in {time.perf_counter() - t0:.0f} s; files in results/: "
          + ", ".join(p.name for p in res["paths"].values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())