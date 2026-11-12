import numpy as np
import pandas as pd
import pytest

from src.respond import policy as P


def frame(n=80, high=(10, 13), gap_on=(5, 80), shift=0.03, forced=None):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    prob = np.zeros(n)
    prob[high[0]:high[1]] = 0.95
    gap = np.zeros(n)
    gap[gap_on[0]:gap_on[1]] = shift
    df = pd.DataFrame({"p_compromise": prob, "gap": gap}, index=idx)
    if forced is not None:
        f = np.zeros(n, bool)
        f[forced[0]:forced[1]] = True
        df["forced"] = f
    return df


def actions(df, **policy):
    return P.decide(df, {"policy": policy})["action"].to_numpy()


def test_off_by_default_keeps_old_rule():
    a = actions(frame())
    assert (a[10:15] == "quarantine").all()
    assert a[15] != "quarantine"


def test_sticky_holds_while_shift_persists():
    a = actions(frame(), sticky=True)
    assert (a[10:] == "quarantine").all()


def test_sticky_releases_when_shift_ends():
    a = actions(frame(gap_on=(5, 13)), sticky=True)
    assert a[20] == "quarantine"
    assert a[30] != "quarantine"


def test_cap_falls_back_to_old_rule():
    a = actions(frame(), sticky=True, max_hold_days=20)
    assert a[-1] != "quarantine"


def test_small_entry_shift_keeps_old_rule():
    a = actions(frame(gap_on=(0, 0), forced=(10, 13), high=(0, 0)), sticky=True)
    assert (a[10:15] == "quarantine").all()
    assert a[15] != "quarantine"


def test_sticky_without_gap_column_keeps_old_rule():
    df = frame().drop(columns="gap")
    a = actions(df, sticky=True)
    assert a[15] != "quarantine"


def test_alert_only_on_entry():
    out = P.decide(frame(), {"policy": {"sticky": True}})
    assert int(out["alert"].sum()) == 1


@pytest.mark.parametrize("bad", [{"release_window": 1}, {"release_frac": 1.0},
                                 {"min_shift": -0.1}, {"max_hold_days": 2}])
def test_invalid_sticky_params(bad):
    with pytest.raises(ValueError):
        P.merge_params(bad)