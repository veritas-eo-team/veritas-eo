"""Tune detector thresholds on train locations only (years before test_year).

python -m src.eval.tune [--quick] [--seeds N]

1. Refuses any frame whose split is not "train".
2. Out-of-fold context-model predictions (ContextModel.oof_predict, GroupKFold by
   location), calibrated per site with calibrate_site; only years before test_year.
3. phi = pooled lag-1 autocorrelation of the calibrated z after the reference year.
4. consistency.clearsky_tol = smallest tolerance broken on <= 0.2% of clean days.
5. k = half the prewhitened z-shift of a 12% step. Grid search of step h, drift L and
   t_thr, replay near_thr and consistency min_violations: each detector gets a share of
   a 1-per-location-year false-alarm budget (measured on the clean out-of-fold years) and
   keeps the value with the best catch rate on the standard attacks injected into the
   same series (then lower false alarms, then lower delay). The whole system (fusion +
   policy) is then checked and the budgets shrink until it stays <= 1 false-alarm
   episode per location-year.
6. Fits the logistic-regression fusion on the same series, saves models/fusion.joblib,
   and compares its location-out-of-fold Brier score with the rules fusion.
7. Writes config/tuned.yaml and prints a before/after table ("before" = settings.yaml
   without any tuned.yaml).

Attacked series reuse the clean calibrated predictions: the model never sees the target,
so an attack only changes gap = log(target + EPS) - log(expected + EPS).
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

from src.attacks.inject import inject
from src.common import EPS, PATHS, get_logger, load_config
from src.detect import consistency, drift, replay, step
from src.detect.context_model import DEFAULT_MIN_REF_DAYS, DEFAULT_REF_DAYS, ContextModel, calibrate_site
from src.detect.fusion import DEFAULT_SLOPE, FEATURE_COLUMNS, fit_fusion, fusion_features, rules_probability, save_fusion
from src.eval.metrics import ALARM_ACTIONS, attack_windows, fa_per_location_year, rising_edges
from src.eval.scenarios import STANDARD, eval_settings
from src.respond.policy import decide

log = get_logger(__name__)

FULL_GRIDS: dict[str, tuple] = {
    "step_h": (4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0),
    "drift_L": (3.0, 3.5, 4.0, 4.5, 5.0, 6.0),
    "drift_t": (3.0, 4.0, 5.0, 6.0, 8.0),
    "replay_near": (0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
    "cons_minv": (1, 2, 3, 4),
}
QUICK_GRIDS: dict[str, tuple] = {
    "step_h": (6.0, 10.0), "drift_L": (4.0, 5.0), "drift_t": (4.0, 6.0),
    "replay_near": (0.8, 0.9), "cons_minv": (2, 3),
}
DEFAULT_BUDGETS: dict[str, float] = {"step": 0.3, "drift": 0.3, "replay": 0.2, "consistency": 0.2}
CATCH_ATTACKS: dict[str, set[str]] = {
    "step": {"step"}, "drift": {"drift"}, "replay": {"replay"}, "consistency": {"step", "drift"},
}
ATTACK_NAMES: tuple[str, ...] = ("step", "drift", "replay")
SYSTEM_FA_TARGET = 1.0
BUDGET_SHRINK = 0.6
MAX_ROUNDS = 6
MAX_EXCEED_RATE = 0.002


# ------------------------------------------------------------------ series

@dataclass
class TuneSeries:
    location: str
    attack: str                  # "none" for clean series
    df: pd.DataFrame             # C1 frame, years before test_year (target possibly attacked)
    gap: pd.Series               # calibrated out-of-fold gap
    z: pd.Series
    sigma: pd.Series
    labels: dict | None
    evaluated: np.ndarray        # days after the calibration reference period
    attacked: np.ndarray
    grace: np.ndarray


def check_train_only(frames: Sequence[pd.DataFrame]) -> None:
    """Tuning must never read test locations."""
    for i, df in enumerate(frames):
        if df.attrs.get("split") != "train":
            name = df.attrs.get("location", f"frame {i}")
            raise ValueError(f"{name} is not a train location; tuning uses train locations only")


def _ref_params(cfg: Mapping[str, Any]) -> tuple[int, int]:
    section = dict(cfg.get("context_model") or {})
    return int(section.get("ref_days") or DEFAULT_REF_DAYS), int(section.get("min_ref_days") or DEFAULT_MIN_REF_DAYS)


def build_clean(frames: Sequence[pd.DataFrame], cfg: Mapping[str, Any]) -> list[TuneSeries]:
    """Out-of-fold, calibrated clean series for each train frame (years before test_year)."""
    check_train_only(frames)
    ty = int(cfg["test_year"])
    ref, min_ref = _ref_params(cfg)
    oof = ContextModel(cfg=cfg).oof_predict(list(frames))
    out = []
    for df, pred in zip(frames, oof):
        keep = np.asarray(df.index.year < ty)
        sub = df.loc[keep]
        cal = calibrate_site(pred.loc[keep], ref, min_ref)
        n = len(sub)
        out.append(TuneSeries(
            location=str(df.attrs.get("location")), attack="none", df=sub,
            gap=cal["gap"], z=cal["z"], sigma=cal["sigma"], labels=None,
            evaluated=np.asarray(sub.index >= sub.index[0] + pd.Timedelta(days=ref)),
            attacked=np.zeros(n, dtype=bool), grace=np.zeros(n, dtype=bool),
        ))
    return out


def build_attacked(clean: Sequence[TuneSeries], settings: Mapping[str, Any], seeds: int = 1,
                   years: Sequence[int] | None = None,
                   attacks: Sequence[str] = ATTACK_NAMES) -> list[TuneSeries]:
    """Standard attacks injected into each clean series, one start per (year, attack, seed)."""
    md_start, md_end = settings["window_start"][5:], settings["window_end"][5:]
    out = []
    for s in clean:
        target = s.df["target"]
        for year in (years or sorted(set(s.df.index.year))[1:]):
            for attack in attacks:
                for seed in range(seeds):
                    attacked, labels = inject(target, attack, seed=seed + 17 * int(year),
                                              window_start=f"{year}-{md_start}", window_end=f"{year}-{md_end}",
                                              **STANDARD[attack])
                    delta = np.log(np.clip(attacked.to_numpy(), 0, None) + EPS) - np.log(np.clip(target.to_numpy(), 0, None) + EPS)
                    gap = s.gap + delta
                    df = s.df.copy()
                    df["target"] = attacked
                    hit, grace = attack_windows(s.df.index, labels)
                    out.append(TuneSeries(location=s.location, attack=attack, df=df, gap=gap, z=gap / s.sigma,
                                          sigma=s.sigma, labels=labels, evaluated=s.evaluated,
                                          attacked=hit, grace=grace))
    return out


# ------------------------------------------------------------------ detector scores (cached)

def _drift_components(s: TuneSeries, p: Mapping[str, Any]) -> dict[str, np.ndarray]:
    zv, gv = s.z.to_numpy(dtype=float), s.gap.to_numpy(dtype=float)
    lam, phi, window = p["lambda"], p["phi"], int(p["slope_window"])
    w = drift.ewma(drift.prewhiten(zv, phi), lam)
    _, t = drift.rolling_slope_t(gv, window, max(10, int(p["min_valid_frac"] * window)))
    return {"w": np.abs(w) / math.sqrt(lam / (2.0 - lam)),
            "t": np.abs(t) / math.sqrt((1.0 + phi) / (1.0 - phi)),
            "invalid": np.isnan(zv) | np.isnan(gv)}


def _drift_score(c: Mapping[str, np.ndarray], L: float, t_thr: float) -> np.ndarray:
    raw = np.maximum(c["w"] / L, np.nan_to_num(c["t"] / t_thr, nan=0.0))
    return pd.Series(np.where(c["invalid"], np.nan, raw)).ffill().fillna(0.0).to_numpy()


def _replay_components(s: TuneSeries, p: Mapping[str, Any]) -> dict[str, np.ndarray]:
    m = p["m"]
    target = s.df["target"]
    x = target.to_numpy(dtype=float)
    n = len(x)
    windows, valid, wstd = replay.window_stats(x, m)
    eligible = valid & (np.nan_to_num(wstd, nan=0.0) >= p["min_window_std"])
    exact, _ = replay.exact_matches(x, m, p["decimals"], eligible)
    dist, _ = replay.nearest_earlier(windows, eligible, m)
    local_std = target.rolling(p["std_window"], min_periods=10).std().to_numpy()
    near = np.zeros(n)
    if len(windows):
        days = np.arange(m - 1, n)
        k = days - m + 1
        ls = local_std[days]
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.clip(1.0 - dist[k] / (math.sqrt(m) * ls), 0.0, 1.0)
        ok = eligible[k] & np.isfinite(dist[k]) & (ls > 0)
        near[days[ok]] = values[ok]
    stuck, _ = replay.stuck_runs(x, p["decimals"], p["stuck_run"])
    return {"near": near, "disagree": replay.rolling_disagreement(s.z.to_numpy(dtype=float), m),
            "exact": exact, "stuck": stuck, "nan": np.isnan(x)}


def _replay_score(c: Mapping[str, np.ndarray], near_thr: float, disagree_thr: float) -> np.ndarray:
    s_near = c["near"] / near_thr * np.minimum(c["disagree"] / disagree_thr, 1.0)
    s_near = np.where(c["disagree"] > disagree_thr, s_near, np.minimum(s_near, replay.NO_DISAGREE_CAP))
    score = np.maximum.reduce([np.where(c["exact"], replay.EXACT_SCORE, 0.0), s_near,
                               np.where(c["stuck"], replay.STUCK_SCORE, 0.0)])
    score[c["nan"]] = 0.0
    return score


def _consistency_score(violation: np.ndarray, window: int, min_violations: int) -> np.ndarray:
    count = pd.Series(violation.astype(float)).rolling(int(window), min_periods=1).sum().to_numpy()
    return count / float(min_violations)


class _Cache:
    """Per-series detector pieces, computed once and reused across the grid."""

    def __init__(self, series: Sequence[TuneSeries]) -> None:
        self.series = list(series)
        self._step: dict[tuple, np.ndarray] = {}
        self._drift: dict[tuple, dict] = {}
        self._replay: dict[tuple, dict] = {}
        self._viol: dict[tuple, np.ndarray] = {}

    def step_score(self, i: int, p: Mapping[str, Any]) -> np.ndarray:
        key = (i, p["phi"], p["k"], p["h"], p["cap"])
        if key not in self._step:
            s = self.series[i]
            self._step[key] = step.detect(s.z, s.gap, p)["score"].to_numpy()
        return self._step[key]

    def drift_comp(self, i: int, p: Mapping[str, Any]) -> dict:
        key = (i, p["phi"], p["lambda"], p["slope_window"], p["min_valid_frac"])
        if key not in self._drift:
            self._drift[key] = _drift_components(self.series[i], p)
        return self._drift[key]

    def replay_comp(self, i: int, p: Mapping[str, Any]) -> dict:
        key = (i, p["m"], p["decimals"], p["min_window_std"], p["std_window"], p["stuck_run"])
        if key not in self._replay:
            self._replay[key] = _replay_components(self.series[i], p)
        return self._replay[key]

    def violation(self, i: int, p: Mapping[str, Any]) -> np.ndarray:
        key = (i, p["clearsky_tol"], p["cloud_thr"], p["clearness_thr"])
        if key not in self._viol:
            over, conflict = consistency.violations(self.series[i].df, p["clearsky_tol"], p["cloud_thr"], p["clearness_thr"])
            self._viol[key] = over | conflict
        return self._viol[key]

    # scores for every series under the full parameter set P
    def scores(self, det: str, P: Mapping[str, Mapping[str, Any]]) -> list[np.ndarray]:
        out = []
        for i in range(len(self.series)):
            if det == "step":
                out.append(self.step_score(i, P["step"]))
            elif det == "drift":
                d = P["drift"]
                out.append(_drift_score(self.drift_comp(i, d), d["L"], d["t_thr"]))
            elif det == "replay":
                r = P["replay"]
                out.append(_replay_score(self.replay_comp(i, r), r["near_thr"], r["disagree_thr"]))
            else:
                c = P["consistency"]
                out.append(_consistency_score(self.violation(i, c), c["window"], c["min_violations"]))
        return out


# ------------------------------------------------------------------ evaluation and selection

def evaluate(alarms: Sequence[np.ndarray], series: Sequence[TuneSeries], attacks: set[str]) -> dict[str, float]:
    """FA episodes per location-year (clean series, after the reference period) and catch
    rate / median delay on attacked series whose attack is in `attacks`."""
    fa_episodes = fa_days = 0
    caught, delays = [], []
    for a, s in zip(alarms, series):
        if s.attack == "none":
            fa_episodes += int(np.sum(rising_edges(a) & s.evaluated))
            fa_days += int(s.evaluated.sum())
        elif s.attack in attacks:
            hits = np.flatnonzero(a & (s.attacked | s.grace))
            caught.append(bool(hits.size))
            if hits.size:
                delays.append(int(hits[0] - np.flatnonzero(s.attacked)[0]))
    return {"fa": fa_per_location_year(fa_episodes, fa_days),
            "catch": float(np.mean(caught)) if caught else float("nan"),
            "delay": float(np.median(delays)) if delays else float("nan")}


def choose(candidates: Sequence[Mapping[str, Any]], budget: float) -> Mapping[str, Any]:
    """Best catch rate within the false-alarm budget, then fewer false alarms, then lower delay.
    If nothing fits the budget, the candidate with the fewest false alarms."""
    feasible = [c for c in candidates if c["fa"] <= budget + 1e-12]
    if not feasible:
        return min(candidates, key=lambda c: (c["fa"], -np.nan_to_num(c["catch"])))
    best = max(round(float(np.nan_to_num(c["catch"])), 2) for c in feasible)
    top = [c for c in feasible if round(float(np.nan_to_num(c["catch"])), 2) == best]
    return min(top, key=lambda c: (c["fa"], c["delay"] if c["delay"] == c["delay"] else math.inf))


def _with(P: Mapping[str, Mapping[str, Any]], det: str, **values: Any) -> dict[str, dict]:
    merge = {"step": step.merge_params, "drift": drift.merge_params,
             "replay": replay.merge_params, "consistency": consistency.merge_params}[det]
    out = {k: dict(v) for k, v in P.items()}
    out[det] = merge({**P[det], **values})
    return out


def system_eval(cache: _Cache, P: Mapping[str, Mapping[str, Any]], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Rules fusion + policy on every series; system FA and catch per attack."""
    slope = float((cfg.get("fusion") or {}).get("slope") or DEFAULT_SLOPE)
    s_step, s_drift = cache.scores("step", P), cache.scores("drift", P)
    s_rep, s_cons = cache.scores("replay", P), cache.scores("consistency", P)
    alarms, probs, dets = [], [], []
    for i, s in enumerate(cache.series):
        comp = cache.replay_comp(i, P["replay"])
        det = pd.DataFrame({"step_score": s_step[i], "drift_score": s_drift[i],
                            "replay_score": s_rep[i], "consistency_score": s_cons[i]}, index=s.df.index)
        forced = comp["exact"] | comp["stuck"]
        p = np.where(forced, np.maximum(rules_probability(det, slope), 0.99), rules_probability(det, slope))
        decided = decide(pd.DataFrame({"p_compromise": p, "forced": forced}, index=s.df.index), cfg)
        alarms.append(decided["action"].isin(ALARM_ACTIONS).to_numpy())
        probs.append(p)
        dets.append(det.assign(replay_exact=comp["exact"], stuck=comp["stuck"]))
    result: dict[str, Any] = {"fa": evaluate(alarms, cache.series, set())["fa"],
                              "probs": probs, "dets": dets}
    for attack in ATTACK_NAMES:
        ev = evaluate(alarms, cache.series, {attack})
        result[f"catch_{attack}"], result[f"delay_{attack}"] = ev["catch"], ev["delay"]
    return result


def fit_lr(cache: _Cache, sys: Mapping[str, Any], models_dir: Path | None) -> dict[str, Any]:
    """Fit and save the LR fusion; compare location-out-of-fold Brier with the rules fusion."""
    X, y, groups, p_rules = [], [], [], []
    for s, det, p in zip(cache.series, sys["dets"], sys["probs"]):
        mask = s.evaluated & ~s.grace
        X.append(fusion_features(det).to_numpy()[mask])
        y.append(s.attacked[mask])
        groups.append(np.full(int(mask.sum()), s.location, dtype=object))
        p_rules.append(p[mask])
    X, y, groups, p_rules = np.vstack(X), np.concatenate(y), np.concatenate(groups), np.concatenate(p_rules)
    model = fit_fusion(X, y)
    path = save_fusion(model, (Path(models_dir) / "fusion.joblib") if models_dir else None)
    p_lr = np.full(len(y), np.nan)
    n_groups = len(np.unique(groups))
    if n_groups >= 2:
        for train, held in GroupKFold(n_splits=min(5, n_groups)).split(X, y, groups):
            if len(np.unique(y[train])) == 2:
                p_lr[held] = fit_fusion(X[train], y[train]).predict_proba(X[held])[:, 1]
    ok = ~np.isnan(p_lr)
    return {"path": path,
            "brier_rules": float(np.mean((p_rules - y) ** 2)),
            "brier_lr_oof": float(np.mean((p_lr[ok] - y[ok]) ** 2)) if ok.any() else float("nan")}


# ------------------------------------------------------------------ main procedure

def _detector_row(cache: _Cache, P: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    row = {}
    for det in ("step", "drift", "replay", "consistency"):
        ev = evaluate([sc >= 1.0 for sc in cache.scores(det, P)], cache.series, CATCH_ATTACKS[det])
        row[f"{det} FA/location-year"], row[f"{det} catch"] = ev["fa"], ev["catch"]
    return row


def _f(x: Any) -> float:
    return round(float(x), 4)


def tune(frames: Sequence[pd.DataFrame], cfg: Mapping[str, Any], *, seeds: int = 1, quick: bool = False,
         out_path: Path | str | None = None, models_dir: Path | str | None = None,
         write: bool = True) -> dict[str, Any]:
    """Run the tuning procedure; returns {params, table, path, lr, budgets}."""
    check_train_only(frames)
    settings = eval_settings(cfg)
    ty = settings["test_year"]
    clean = build_clean(frames, cfg)
    first_year = min(int(s.df.index.year.min()) for s in clean)
    years = list(range(first_year + 1, ty))
    if quick:
        years = years[-2:]
    series = clean + build_attacked(clean, settings, seeds=1 if quick else seeds, years=years)
    cache = _Cache(series)
    grids = QUICK_GRIDS if quick else FULL_GRIDS
    log.info("tuning on %d clean and %d attacked series (years %s-%s)",
             len(clean), len(series) - len(clean), years[0], years[-1])

    base = {"step": step.params_from_config(cfg), "drift": drift.params_from_config(cfg),
            "replay": replay.params_from_config(cfg), "consistency": consistency.params_from_config(cfg)}
    phi = step.estimate_phi([s.z.to_numpy()[s.evaluated] for s in clean])
    sigma = np.concatenate([s.sigma.to_numpy()[s.evaluated] for s in clean])
    shift = math.log(1.12) / float(np.nanmedian(sigma))
    k = float(np.clip(0.5 * shift * (1.0 - phi) / math.sqrt(1.0 - phi * phi), 0.25, 1.5))
    tol = consistency.choose_clearsky_tol([s.df for s in clean], MAX_EXCEED_RATE)
    start = _with(_with(_with(base, "step", phi=phi, k=k), "drift", phi=phi), "consistency", clearsky_tol=tol)

    def candidates(det: str, grid: Sequence[Any], setter) -> list[dict[str, Any]]:
        out = []
        for value in grid:
            P = setter(value)
            ev = evaluate([sc >= 1.0 for sc in cache.scores(det, P)], series, CATCH_ATTACKS[det])
            out.append({"value": value, **ev})
        return out

    cands = {
        "step": candidates("step", grids["step_h"], lambda h: _with(start, "step", h=h)),
        "drift": candidates("drift", [(L, t) for L in grids["drift_L"] for t in grids["drift_t"]],
                            lambda v: _with(start, "drift", L=v[0], t_thr=v[1])),
        "replay": candidates("replay", grids["replay_near"], lambda n: _with(start, "replay", near_thr=n)),
        "consistency": candidates("consistency", grids["cons_minv"],
                                  lambda m: _with(start, "consistency", min_violations=m)),
    }

    budgets = dict(DEFAULT_BUDGETS)
    for _ in range(MAX_ROUNDS):
        pick = {det: choose(c, budgets[det]) for det, c in cands.items()}
        tuned = _with(start, "step", h=pick["step"]["value"])
        tuned = _with(tuned, "drift", L=pick["drift"]["value"][0], t_thr=pick["drift"]["value"][1])
        tuned = _with(tuned, "replay", near_thr=pick["replay"]["value"])
        tuned = _with(tuned, "consistency", min_violations=pick["consistency"]["value"])
        after = system_eval(cache, tuned, cfg)
        if after["fa"] <= SYSTEM_FA_TARGET:
            break
        budgets = {det: b * BUDGET_SHRINK for det, b in budgets.items()}
    before = system_eval(cache, base, cfg)
    lr = fit_lr(cache, after, Path(models_dir) if models_dir else None)

    rows = []
    for name, P, sysres in (("before", base, before), ("after", tuned, after)):
        row = _detector_row(cache, P)
        row["system FA/location-year"] = sysres["fa"]
        for attack in ATTACK_NAMES:
            row[f"system catch {attack}"] = sysres[f"catch_{attack}"]
            row[f"system median delay {attack} (days)"] = sysres[f"delay_{attack}"]
        rows.append(pd.Series(row, name=name))
    table = pd.concat(rows, axis=1).reset_index().rename(columns={"index": "metric"})

    params = {
        "step": {"phi": _f(phi), "k": _f(k), "h": _f(tuned["step"]["h"])},
        "drift": {"phi": _f(phi), "L": _f(tuned["drift"]["L"]), "t_thr": _f(tuned["drift"]["t_thr"])},
        "replay": {"near_thr": _f(tuned["replay"]["near_thr"])},
        "consistency": {"clearsky_tol": _f(tol), "min_violations": int(tuned["consistency"]["min_violations"])},
    }
    record = {
        **params,
        "tuning": {
            "train_locations": sorted({s.location for s in clean}),
            "years": [int(years[0]), int(years[-1])],
            "clean_location_years": _f(sum(s.evaluated.sum() for s in clean) / 365.25),
            "attacked_series": int(len(series) - len(clean)),
            "fa_target_per_location_year": SYSTEM_FA_TARGET,
            "budgets": {d: _f(b) for d, b in budgets.items()},
            "system_before": {"fa_per_location_year": _f(before["fa"]),
                              **{f"catch_{a}": _f(before[f"catch_{a}"]) for a in ATTACK_NAMES}},
            "system_after": {"fa_per_location_year": _f(after["fa"]),
                             **{f"catch_{a}": _f(after[f"catch_{a}"]) for a in ATTACK_NAMES}},
            "fusion_brier_rules": _f(lr["brier_rules"]),
            "fusion_brier_lr_location_oof": _f(lr["brier_lr_oof"]),
        },
    }
    path = Path(out_path) if out_path is not None else PATHS["tuned"]
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# written by: python -m src.eval.tune (train locations only, years before test_year)\n")
            yaml.safe_dump(record, fh, sort_keys=False)
    return {"params": params, "table": table, "path": path, "lr": lr, "budgets": budgets}


def main(argv: Sequence[str] | None = None) -> int:
    from src.data.align import list_locations, load

    parser = argparse.ArgumentParser(description="Tune thresholds on train locations (years before test_year).")
    parser.add_argument("--quick", action="store_true", help="small grids, last 2 years only")
    parser.add_argument("--seeds", type=int, default=1, help="attack starts per (location, year, attack)")
    args = parser.parse_args(argv)

    # "before" = settings.yaml alone, ignoring any earlier tuned.yaml
    cfg = load_config(tuned_path=PATHS["config"] / "_ignore_tuned_.yaml")
    frames = [load(location_id, cfg) for location_id in list_locations("train", cfg)]
    res = tune(frames, cfg, seeds=args.seeds, quick=args.quick)
    print(res["table"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"tuned: {res['params']}")
    print(f"fusion Brier: rules {res['lr']['brier_rules']:.4f}, LR (location out-of-fold) {res['lr']['brier_lr_oof']:.4f}")
    print(f"wrote config/{res['path'].name} and models/{res['lr']['path'].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())