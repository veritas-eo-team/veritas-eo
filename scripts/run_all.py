"""One-command run: download -> align -> train -> tune -> benchmark -> verify the log.

python scripts/run_all.py [--offline] [--quick] [--force]

Finished steps are skipped:
  download   NASA POWER cells already cached are never fetched again; Sentinel-2 only if
             CDSE_CLIENT_ID / CDSE_CLIENT_SECRET are set (optional)
  align      rebuilds a location only if an input file or a setting changed
  train      skipped if models/context_model.joblib loads and is newer than the clean data
  tune       skipped if config/tuned.yaml is newer than the model
  benchmark  skipped if results/summary.md is newer than the model and tuned.yaml
  verify     always runs (signed hash-chained provenance log)
--offline  no network: uses cached data only (fails if a location was never downloaded)
--quick    quick tuning and a quick benchmark (a few minutes instead of ~15)
--force    re-run train, tune and benchmark even if they look finished
Each step runs its own CLI (python -m ...), so the output is the same as running it by hand.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import PATHS, load_config  # noqa: E402


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _is_newer(path: Path, *others: Path) -> bool:
    """True if `path` exists and is at least as new as every existing path in `others`."""
    if not path.exists():
        return False
    times = [_mtime(o) for o in others if o.exists()]
    return not times or _mtime(path) >= max(times)


def _run(module: str, *args: str) -> int:
    print(f"\n$ python -m {module} {' '.join(args)}".rstrip(), flush=True)
    return subprocess.call([sys.executable, "-m", module, *args], cwd=ROOT)


def step_download(offline: bool) -> tuple[str, int]:
    from src.data import power
    from src.data.align import list_locations

    cfg = load_config()
    if offline:
        missing = [loc for loc in list_locations(cfg=cfg) if not power.cell_path(loc, power.SITE_CELL).exists()]
        if missing:
            print(f"offline: no cached NASA POWER data for {missing}; run once without --offline")
            return "failed", 1
        print("offline: using cached NASA POWER data")
        return "skipped", 0
    code = _run("src.data.power", "--all")
    if code:
        return "failed", code
    if os.environ.get("CDSE_CLIENT_ID") and os.environ.get("CDSE_CLIENT_SECRET"):
        if _run("src.data.sentinel2", "--all"):
            print("Sentinel-2 download had failures; continuing (the system works without it)")
    else:
        print("Sentinel-2: CDSE_CLIENT_ID / CDSE_CLIENT_SECRET not set; using cached data if present")
    return "ran", 0


def step_align() -> tuple[str, int]:
    code = _run("src.data.align", "--all")
    return ("ran" if code == 0 else "failed"), code


def step_train(force: bool) -> tuple[str, int]:
    model = PATHS["models"] / "context_model.joblib"
    cleans = sorted(PATHS["clean"].glob("*.parquet"))
    if not force and _is_newer(model, *cleans):
        try:
            from src.detect.context_model import ContextModel
            ContextModel.load(model)
            print("context model is up to date; skipping training")
            return "skipped", 0
        except (ValueError, FileNotFoundError) as exc:
            print(f"saved model cannot be used ({exc}); retraining")
    code = _run("src.detect.context_model", "--train")
    return ("ran" if code == 0 else "failed"), code


def step_tune(force: bool, quick: bool) -> tuple[str, int]:
    model = PATHS["models"] / "context_model.joblib"
    if not force and _is_newer(PATHS["tuned"], model):
        print("config/tuned.yaml is newer than the model; skipping tuning")
        return "skipped", 0
    code = _run("src.eval.tune", *(["--quick"] if quick else []))
    return ("ran" if code == 0 else "failed"), code


def step_benchmark(force: bool, quick: bool) -> tuple[str, int]:
    summary = PATHS["results"] / "summary.md"
    model = PATHS["models"] / "context_model.joblib"
    if not force and _is_newer(summary, model, PATHS["tuned"]):
        print("results/summary.md is newer than the model and tuned.yaml; skipping the benchmark")
        return "skipped", 0
    code = _run("src.eval.run_benchmark", *(["--quick", "--sweep"] if quick else ["--sweep"]))
    return ("ran" if code == 0 else "failed"), code


def step_verify() -> tuple[str, int]:
    code = _run("src.trust.verify", "--allow-missing")
    return ("ran" if code == 0 else "failed"), code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the whole pipeline end to end.")
    parser.add_argument("--offline", action="store_true", help="no network; use cached data only")
    parser.add_argument("--quick", action="store_true", help="quick tuning and benchmark")
    parser.add_argument("--force", action="store_true", help="re-run train, tune and benchmark")
    args = parser.parse_args(argv)

    steps = [
        ("download", lambda: step_download(args.offline)),
        ("align", step_align),
        ("train", lambda: step_train(args.force)),
        ("tune", lambda: step_tune(args.force, args.quick)),
        ("benchmark", lambda: step_benchmark(args.force, args.quick)),
        ("verify log", step_verify),
    ]
    report, exit_code = [], 0
    t_start = time.perf_counter()
    for name, fn in steps:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        status, code = fn()
        report.append((name, status, time.perf_counter() - t0))
        if code:
            exit_code = code
            print(f"step '{name}' failed (exit code {code}); stopping")
            break
    print("\n=== summary ===")
    for name, status, seconds in report:
        print(f"{name:<12} {status:<8} {seconds:7.1f} s")
    print(f"total {time.perf_counter() - t_start:.0f} s; results in results/ (summary.md, benchmark_table.csv, charts)")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())