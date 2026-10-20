"""Shared pytest fixtures: cfg, frame and tmp_paths."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.common import PATHS, build_paths, load_config  # noqa: E402
from src.synthetic import make_synthetic_frame  # noqa: E402

_REAL_SETTINGS = _ROOT / "config" / "settings.yaml"


@pytest.fixture
def cfg() -> dict:
    """The merged project config."""
    return load_config()


@pytest.fixture
def frame():
    """A 2018-2024 synthetic C1 frame with 4 neighbours (seed 0)."""
    return make_synthetic_frame(seed=0)


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    """Redirect every PATHS entry into tmp_path, with a copy of settings.yaml."""
    for key, path in build_paths(tmp_path).items():
        monkeypatch.setitem(PATHS, key, path)
    shutil.copyfile(_REAL_SETTINGS, PATHS["settings"])
    return PATHS
