import numpy as np
import pytest

from src.common import load_config
from src.data.align import load
from src.detect.context_model import ContextModel
from src.eval import red_team as RT


@pytest.fixture(scope="module")
def oracle():
    cfg = load_config()
    o = RT.clean_oracle(load("kairouan", cfg), ContextModel.load(), cfg, horizon=30)
    assert o is not None
    return o


def test_clean_run_has_no_alarm(oracle):
    assert oracle.first_alarm(1.0) is None


def test_big_step_is_detected(oracle):
    fa = oracle.first_alarm(1.12)
    assert fa is not None and fa < 30


def test_oracle_does_not_change_the_data(oracle):
    before = oracle.clean.copy()
    oracle.first_alarm(1.3)
    assert np.array_equal(before, oracle.clean)


def test_envelope_is_between_zero_and_twelve_percent(oracle):
    size = RT.max_undetected(oracle, 30, steps=5)
    assert 0.0 <= size < 0.12


def test_greedy_bias_stays_hidden(oracle):
    bias, caught = RT.greedy_attack(oracle, 14, steps=4)
    assert len(bias) == 14 and caught is None
    assert (bias >= 0).all() and (bias <= RT.MAX_SIZE).all()
    assert oracle.first_alarm(1.0 + np.concatenate([bias, np.zeros(16)])) is None