import numpy as np
import pandas as pd
import pytest

from src.detect import consistency as C


def make_frame(n=600, attack_start=None, factor=0.92, seed=0, with_cloud=True):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    clr = np.full(n, 30.0)
    clearness = np.full(n, 0.6)
    cloud = np.full(n, np.nan)
    passes = np.arange(0, n, 5)
    cloud[passes] = rng.choice([0.0, 0.05, 0.9], size=len(passes), p=[0.5, 0.2, 0.3])
    clear = cloud <= 0.1
    clearness[clear] = 0.76 + rng.normal(0, 0.01, int(clear.sum()))
    target = clr * clearness
    if attack_start is not None:
        target[attack_start:] *= factor
    cols = {"target": target, C.CLRSKY_COL: clr}
    if with_cloud:
        cols["cloud"] = cloud
    return pd.DataFrame(cols, index=idx)


def test_mirror_off_by_default_keeps_old_output():
    df = make_frame(attack_start=450)
    out = C.detect(df)
    assert "under_clearsky" not in out.columns
    assert list(out.columns) == ["score", "alarm", "over_clearsky", "cloud_conflict"]


def test_mirror_catches_downward_step():
    start = 450
    out = C.detect(make_frame(attack_start=start), params={"mirror": True})
    assert not out["alarm"].iloc[:start].any()
    hits = np.flatnonzero(out["alarm"].to_numpy())
    assert hits.size and start <= hits[0] <= start + 60
    assert out.attrs["first_alarm"] is not None
    assert out.attrs["est_start"] >= out.index[start]


def test_mirror_quiet_on_clean_data():
    out = C.detect(make_frame(), params={"mirror": True})
    assert not out["alarm"].any()


def test_mirror_inactive_without_cloud_column():
    out = C.detect(make_frame(attack_start=450, with_cloud=False), params={"mirror": True})
    assert not out["under_clearsky"].any()
    assert not out["alarm"].any()


def test_mirror_ignores_upward_step():
    out = C.detect(make_frame(attack_start=450, factor=1.05), params={"mirror": True})
    assert not out["under_clearsky"].any()


def test_reference_is_causal():
    df = make_frame(attack_start=450)
    p = C.merge_params({"mirror": True})
    full = C.under_clearsky(df, p)
    cut = C.under_clearsky(df.iloc[:500], p)
    assert np.array_equal(full[:500], cut)


def test_under_natural_rate_low_on_clean_data():
    rates = C.under_natural_rate([make_frame(seed=s) for s in range(3)], params={"mirror": True})
    assert rates.loc[0.05, "rate"] <= 0.02
    assert C.choose_under_tol([make_frame(seed=s) for s in range(3)]) <= 0.05


@pytest.mark.parametrize("bad", [{"ref_lag": 400, "ref_span": 365}, {"under_tol": 0.0},
                                 {"clear_cloud_thr": 0.8}, {"under_window": 0}])
def test_invalid_mirror_params(bad):
    with pytest.raises(ValueError):
        C.merge_params(bad)