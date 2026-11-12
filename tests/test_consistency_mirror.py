import numpy as np
import pandas as pd
import pytest

from src.detect import consistency as C


def make_frame(n=700, attack_start=None, factor=0.94, seed=0, with_cloud=True, s2_always_cloudy=False):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    clr = np.full(n, 30.0)
    clear = rng.random(n) < 0.6
    nb_cl = np.where(clear, 0.76, 0.45)[:, None] + rng.normal(0, 0.01, (n, 4))
    nb = clr[:, None] * nb_cl
    target = 1.02 * nb.mean(axis=1) * (1 + rng.normal(0, 0.01, n))
    if attack_start is not None:
        target[attack_start:] *= factor
    cols = {"target": target, **{f"nb_{i}": nb[:, i] for i in range(4)}, C.CLRSKY_COL: clr}
    if with_cloud:
        cloud = np.full(n, np.nan)
        passes = np.arange(0, n, 5)
        cloud[passes] = 0.9 if s2_always_cloudy else np.where(clear[passes], 0.05, 0.9)
        cols["cloud"] = cloud
    return pd.DataFrame(cols, index=idx)


def test_mirror_off_by_default_keeps_old_output():
    out = C.detect(make_frame(attack_start=500))
    assert list(out.columns) == ["score", "alarm", "over_clearsky", "cloud_conflict"]


def test_mirror_catches_downward_step():
    start = 500
    out = C.detect(make_frame(attack_start=start), params={"mirror": True})
    assert not out["alarm"].iloc[:start].any()
    hits = np.flatnonzero(out["alarm"].to_numpy())
    assert hits.size and start <= hits[0] <= start + 40
    assert out.attrs["est_start"] is not None


def test_mirror_quiet_on_clean_data():
    for seed in range(3):
        out = C.detect(make_frame(seed=seed), params={"mirror": True})
        assert not out["alarm"].any()


def test_mirror_works_without_sentinel2():
    start = 500
    out = C.detect(make_frame(attack_start=start, with_cloud=False), params={"mirror": True})
    hits = np.flatnonzero(out["alarm"].to_numpy())
    assert hits.size and start <= hits[0] <= start + 40


def test_sentinel2_veto_removes_clear_days():
    df = make_frame(attack_start=500, s2_always_cloudy=True)
    clear = C.clear_days(df, 0.7, 0.3)
    assert not clear[np.arange(0, len(df), 5)].any()


def test_mirror_ignores_upward_step():
    out = C.detect(make_frame(attack_start=500, factor=1.05), params={"mirror": True})
    assert not out["under_clearsky"].any()
    assert not out["alarm"].any()


def test_mirror_is_causal():
    df = make_frame(attack_start=500)
    p = C.merge_params({"mirror": True})
    full = C.mirror_score(C.mirror_deviation(df, p), p)
    cut = C.mirror_score(C.mirror_deviation(df.iloc[:550], p), p)
    assert np.allclose(full[:550], cut)


def test_clear_days_ignore_the_target():
    df = make_frame()
    a = C.clear_days(df, 0.7, 0.3)
    df2 = df.copy()
    df2["target"] *= 0.5
    assert np.array_equal(a, C.clear_days(df2, 0.7, 0.3))


def test_natural_alarms_low_on_clean_data():
    frames = [make_frame(seed=s) for s in range(3)]
    rates = C.mirror_natural_alarms(frames)
    assert rates.loc[0.04, "episodes_per_year"] == 0
    assert C.choose_under_tol(frames) <= 0.04


@pytest.mark.parametrize("bad", [{"ref_lag": 400, "ref_span": 365}, {"under_tol": 0.0},
                                 {"clear_cloud_thr": 0.8}, {"under_n": 0}, {"nb_clear_thr": 0.0}])
def test_invalid_mirror_params(bad):
    with pytest.raises(ValueError):
        C.merge_params(bad)