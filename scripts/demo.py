"""30-second terminal demo.

python scripts/demo.py [--location ID] [--fast]

1. Injects a 12% step bias into a test location (the standard attack) and shows, day by
   day, the observed value, the expected value from untouched context, the gap, the trust
   score and the action. The alert line marks the start of quarantine.
2. Runs the provenance tamper demo: edits a temporary copy of a logged file and shows that
   verification catches it (real data is never touched).
Uses the real data and trained model if present, otherwise synthetic data.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.attacks.inject import inject  # noqa: E402
from src.common import load_config  # noqa: E402
from src.detect.context_model import ContextModel  # noqa: E402
from src.pipeline import ALARM_ACTIONS, run_pipeline  # noqa: E402


def load_inputs(location: str | None, cfg: dict) -> tuple[pd.DataFrame, ContextModel, str]:
    """Real clean frame + trained model, or synthetic data with a quickly fitted model."""
    try:
        from src.data.align import list_locations, load
        location = location or list_locations("test", cfg)[0]
        model = ContextModel.load()
        return load(location, cfg), model, f"{location} (NASA POWER, unseen test location)"
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[real data or model not available: {exc}]\n[using synthetic data instead]\n")
    from src.synthetic import make_synthetic_frame
    frames = [make_synthetic_frame(location=f"train{i}", seed=i) for i in range(3)]
    model = ContextModel(cfg=cfg).fit(frames)
    return make_synthetic_frame(location="demo", seed=7, split="test"), model, "synthetic location"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="30-second terminal demo.")
    parser.add_argument("--location", default=None, help="test location (default: the first one)")
    parser.add_argument("--fast", action="store_true", help="no pause between lines")
    args = parser.parse_args(argv)
    pause = 0.0 if args.fast else 0.12

    cfg = load_config()
    df, model, label = load_inputs(args.location, cfg)
    ty = int(cfg["test_year"])
    start = pd.Timestamp(f"{ty}-05-01")
    attacked, labels = inject(df["target"], "step", size=0.12, start=start)
    run_df = df.loc[start - pd.Timedelta(days=365): start + pd.Timedelta(days=60)].copy()
    run_df["target"] = attacked.loc[run_df.index]

    print(f"Attack: +12% step bias from {start.date()} on {label}")
    print("The detector only sees the series; it is not told when the attack starts.\n")
    out = run_pipeline(run_df, model, cfg)
    alarm = out["action"].isin(ALARM_ACTIONS)
    after = alarm.loc[start:]
    first = after[after].index[0] if after.any() else None
    end = max(start + pd.Timedelta(days=20), (first or start) + pd.Timedelta(days=8))
    view = out.loc[start - pd.Timedelta(days=5): end]

    print(f"{'date':<11}{'':2}{'observed':>9}{'expected':>9}{'gap':>8}{'trust':>7}  action")
    for day, row in view.iterrows():
        mark = "*" if day >= start else " "
        gap_pct = (np.exp(row["gap"]) - 1.0) * 100 if row["gap"] == row["gap"] else float("nan")
        line = (f"{day.date()!s:<11}{mark:2}{row['target']:9.2f}{row['expected']:9.2f}"
                f"{gap_pct:+7.1f}%{row['trust']:7.2f}  {row['action']}")
        if row["alert"]:
            line += "   <<< ALERT: quarantine starts"
        print(line, flush=True)
        time.sleep(pause)
    print("(* = attacked day; gap = observed vs expected from neighbours and other variables)\n")

    if first is not None:
        row = out.loc[first]
        est = row["est_start"]
        print(f"Detected on {first.date()}: {(first - start).days} day(s) after the attack began; "
              f"type '{row['attack_type']}', estimated start {est.date() if not pd.isna(est) else '-'}.")
    else:
        print("No alarm in the shown window.")

    print("\nTamper-proof log: editing a logged file (on a temporary copy) ...")
    try:
        from src.trust.verify import tamper_demo
        res = tamper_demo()
        if not res.ok:
            print(f"CAUGHT at record {res.first_bad_seq}: {res.reason}")
        else:
            print("Not caught (unexpected); check the provenance log.")
    except Exception as exc:  # the demo should never crash on this part
        print(f"tamper demo unavailable here: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())