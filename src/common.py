"""Shared paths, configuration, logging and reproducibility helpers."""
from __future__ import annotations

import copy
import logging
import os
import random
import subprocess
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

ROOT: Path = Path(__file__).resolve().parents[1]

EPS: float = 0.01
"""Offset used in log(x + EPS). Must equal `eps` in settings.yaml (checked by tests)."""

_DIR_LAYOUT: dict[str, str] = {
    "root": ".",
    "config": "config",
    "data": "data",
    "raw": "data/raw",
    "raw_power": "data/raw/power",
    "raw_s2": "data/raw/s2",
    "clean": "data/clean",
    "provenance": "data/provenance",
    "keys": "keys",
    "models": "models",
    "results": "results",
    "report": "report",
    "src": "src",
    "app": "app",
    "scripts": "scripts",
    "tests": "tests",
    "fixtures": "tests/fixtures",
}

_FILE_LAYOUT: dict[str, str] = {
    "settings": "config/settings.yaml",
    "tuned": "config/tuned.yaml",
    "provenance_log": "data/provenance/log.jsonl",
    "readings_log": "data/provenance/readings.jsonl",
}

_REQUIRED_KEYS: tuple[str, ...] = (
    "target", "context_vars", "start", "end", "test_year", "eps",
    "neighbor_offset_deg", "locations", "context_model", "step", "drift",
    "replay", "consistency", "fusion", "policy", "eval",
)
_LOCATION_KEYS: tuple[str, ...] = ("id", "lat", "lon", "climate", "split")
_SPLITS: tuple[str, ...] = ("train", "test")
_FUSION_MODES: tuple[str, ...] = ("rules", "lr")


def build_paths(root: Path | str) -> dict[str, Path]:
    """Map every layout key (folders and fixed files) to a path under `root`."""
    root = Path(root)
    paths: dict[str, Path] = {}
    for key, rel in {**_DIR_LAYOUT, **_FILE_LAYOUT}.items():
        paths[key] = root if rel == "." else root / rel
    return paths


class _AutoPaths(dict):
    """Dict whose item access creates the folder (or a file's parent folder).

    Use PATHS["key"]; .get() and iteration do not create folders.
    """

    def __getitem__(self, key: str) -> Path:
        path = Path(super().__getitem__(key))
        folder = path.parent if key in _FILE_LAYOUT else path
        folder.mkdir(parents=True, exist_ok=True)
        return path


PATHS: dict[str, Path] = _AutoPaths(build_paths(ROOT))


def rel_to_root(path: Path | str) -> str:
    """Return `path` relative to the project root as a POSIX string."""
    root = Path(PATHS["root"]).resolve()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        raise ValueError(f"{resolved.name} is outside the project root") from None


def from_root(rel: str) -> Path:
    """Inverse of rel_to_root: project-relative POSIX string to a Path."""
    return Path(PATHS["root"]) / rel


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base`. Lists are replaced."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping; an empty file gives {}."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: top level must be a mapping")
    return data


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Raise ValueError if the merged config is incomplete or inconsistent."""
    missing = [key for key in _REQUIRED_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"config is missing keys: {missing}")

    start = date.fromisoformat(str(cfg["start"]))
    end = date.fromisoformat(str(cfg["end"]))
    if end <= start:
        raise ValueError("config end must be after start")
    if not start.year <= int(cfg["test_year"]) <= end.year:
        raise ValueError("test_year must lie between start and end")
    if float(cfg["eps"]) <= 0:
        raise ValueError("eps must be positive")

    seen: set[str] = set()
    for loc in cfg["locations"]:
        absent = [key for key in _LOCATION_KEYS if key not in loc]
        if absent:
            raise ValueError(f"location entry is missing {absent}")
        if loc["id"] in seen:
            raise ValueError(f"duplicate location id: {loc['id']}")
        seen.add(loc["id"])
        if loc["split"] not in _SPLITS:
            raise ValueError(f"{loc['id']}: split must be one of {_SPLITS}")
        if not -90.0 <= float(loc["lat"]) <= 90.0:
            raise ValueError(f"{loc['id']}: latitude out of range")
        if not -180.0 <= float(loc["lon"]) <= 180.0:
            raise ValueError(f"{loc['id']}: longitude out of range")

    policy = cfg["policy"]
    if not 1.0 >= policy["accept"] > policy["deweight"] > policy["verify"] >= 0.0:
        raise ValueError("policy thresholds must satisfy 1 >= accept > deweight > verify >= 0")
    if cfg["fusion"].get("mode") not in _FUSION_MODES:
        raise ValueError(f"fusion.mode must be one of {_FUSION_MODES}")


def load_config(
    settings_path: Path | str | None = None,
    tuned_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load settings.yaml, deep-merged with tuned.yaml if present, and validate.

    tuned.yaml may override any section except `locations`, so tuning can never
    change the train/test split.
    """
    settings_path = Path(settings_path) if settings_path is not None else PATHS["settings"]
    tuned_path = Path(tuned_path) if tuned_path is not None else PATHS["tuned"]
    if not settings_path.exists():
        raise FileNotFoundError(f"missing config file: {settings_path.name}")

    cfg = _read_yaml(settings_path)
    if tuned_path.exists():
        tuned = _read_yaml(tuned_path)
        if "locations" in tuned:
            raise ValueError("tuned.yaml must not override locations (the split is fixed)")
        cfg = deep_merge(cfg, tuned)

    for key in ("start", "end"):
        if key in cfg:
            cfg[key] = str(cfg[key])
    validate_config(cfg)
    return cfg


def get_location(cfg: Mapping[str, Any], location_id: str) -> dict[str, Any]:
    """Return a copy of the config entry for `location_id`."""
    for loc in cfg["locations"]:
        if loc["id"] == location_id:
            return dict(loc)
    raise KeyError(f"unknown location: {location_id}")


_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_logging_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; configure a single console handler once.

    The level comes from the LOG_LEVEL environment variable (default INFO).
    The format has no file paths, so logs stay anonymous.
    """
    global _logging_configured
    if not _logging_configured:
        root = logging.getLogger()
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(_LOG_FORMAT, "%H:%M:%S"))
            root.addHandler(handler)
        level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
        root.setLevel(level if isinstance(level, int) else logging.INFO)
        _logging_configured = True
    return logging.getLogger(name)


@lru_cache(maxsize=1)
def code_version() -> str:
    """Short git commit hash of the repository, or "unknown"."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    version = out.stdout.strip()
    return version or "unknown"


def set_seed(seed: int) -> np.random.Generator:
    """Seed Python and NumPy global RNGs; return a fresh Generator with the same seed."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)
