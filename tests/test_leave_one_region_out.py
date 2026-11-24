"""Tests for the leave-one-regime-out evaluation helpers (offline)."""
import pandas as pd

from scripts.leave_one_region_out import REGIMES, summarize
from src.common import load_config
from src.data.align import list_locations


def test_regimes_cover_exactly_the_train_locations():
    cfg = load_config()
    train = set(list_locations("train", cfg))
    test = set(list_locations("test", cfg))
    covered = {loc for held in REGIMES.values() for loc in held}
    assert covered == train
    assert not (covered & test)


def test_summarize_catch_delay_and_false_alarms():
    records = pd.DataFrame([
        {"regime": "cold", "attack": "none", "caught": False, "delay_days": float("nan"), "fa_episodes": 2},
        {"regime": "cold", "attack": "none", "caught": False, "delay_days": float("nan"), "fa_episodes": 0},
        {"regime": "cold", "attack": "step", "caught": True, "delay_days": 2.0, "fa_episodes": 0},
        {"regime": "cold", "attack": "step", "caught": False, "delay_days": float("nan"), "fa_episodes": 0},
        {"regime": "cold", "attack": "drift", "caught": True, "delay_days": 8.0, "fa_episodes": 0},
    ])
    table = summarize(records).set_index(["regime", "attack"])
    assert table.loc[("cold", "step"), "catch_rate"] == 0.5
    assert table.loc[("cold", "step"), "median_delay_days"] == 2.0
    assert table.loc[("cold", "step"), "fa_per_location_year"] == 1.0
    assert table.loc[("cold", "none"), "fa_per_location_year"] == 1.0
    assert pd.isna(table.loc[("cold", "none"), "catch_rate"])
