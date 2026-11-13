import numpy as np
import pandas as pd
import pytest

from src.common import load_config
from src.data.align import load
from src.eval.baselines import Baselines, as_c5
from src.pipeline import C5_COLUMNS


@pytest.fixture(scope="module")
def setup():
    cfg = load_config()
    train = {i: load(i, cfg) for i in ("tunis", "tozeur", "seville")}
    return cfg, train, Baselines(cfg, train)


def test_as_c5_has_contract_columns():
    df = pd.DataFrame({"target": np.full(10, 20.0)}, index=pd.date_range("2024-01-01", periods=10, name="date"))
    out = as_c5(df, np.r_[np.zeros(5), np.full(5, 3.0)], 1.0)
    assert list(out.columns) == C5_COLUMNS
    assert (out["action"].iloc[:5] == "accept").all() and (out["action"].iloc[5:] == "quarantine").all()
    assert int(out["alert"].sum()) == 1


@pytest.mark.parametrize("kind", ["zscore", "cusum", "iforest"])
def test_scores_are_causal(setup, kind):
    cfg, train, base = setup
    df = train["tunis"].loc["2022-01-01":"2023-12-31"]
    full, _ = base.score(kind, df)
    cut, _ = base.score(kind, df.iloc[:500])
    assert np.allclose(np.nan_to_num(full[:500]), np.nan_to_num(cut))


@pytest.mark.parametrize("kind", ["zscore", "cusum", "iforest"])
def test_tuning_respects_budget(setup, kind):
    cfg, train, base = setup
    thr, rate = base.tune(kind, train, 1.0)
    assert rate <= 1.0 and thr > 0


def test_pipelines_return_c5(setup):
    cfg, train, base = setup
    from src.detect.context_model import ContextModel
    model = ContextModel.load()
    df = train["tunis"].loc["2023-01-01":"2024-12-31"]
    base.tune("zscore", train, 1.0)
    for kind in ("veritas_no_nb", "zscore"):
        out = base.pipeline(kind)(df, model, cfg)
        assert list(out.columns) == C5_COLUMNS and len(out) == len(df)