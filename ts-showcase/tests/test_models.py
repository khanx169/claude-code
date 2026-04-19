"""Tests for tsshowcase.models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsshowcase.models import (
    bottom_up_hierarchical,
    fit_ets,
    fit_naive,
    fit_seasonal_naive,
)

# ── Shared synthetic fixture ───────────────────────────────────────────────────

_N = 32
_QUARTERS = pd.period_range("2000Q1", periods=_N, freq="Q")
_rng = np.random.default_rng(42)
_SERIES: pd.Series = pd.Series(
    100.0
    + np.arange(_N) * 0.5
    + 10.0 * np.sin(np.arange(_N) * np.pi / 2)
    + _rng.normal(0, 0.5, _N),
    index=_QUARTERS,
    name="synthetic",
)
_HORIZON = 8
_CACHE = Path("data/tourism.csv")


@pytest.fixture(scope="module")
def tourism_train() -> pd.DataFrame:
    if not _CACHE.exists():
        pytest.skip("data/tourism.csv not present — skipping integration tests")
    from tsshowcase.data import load_tourism, split_train_test

    df = load_tourism(_CACHE)
    train, _ = split_train_test(df)
    return train


# ── fit_naive ─────────────────────────────────────────────────────────────────


def test_naive_length() -> None:
    fc = fit_naive(_SERIES, _HORIZON)
    assert len(fc) == _HORIZON


def test_naive_period_index() -> None:
    fc = fit_naive(_SERIES, _HORIZON)
    assert isinstance(fc.index, pd.PeriodIndex)
    assert fc.index[0] == _SERIES.index[-1] + 1


def test_naive_all_equal_last() -> None:
    fc = fit_naive(_SERIES, _HORIZON)
    assert (fc == _SERIES.iloc[-1]).all()


def test_naive_all_finite() -> None:
    fc = fit_naive(_SERIES, _HORIZON)
    assert np.isfinite(fc.values).all()


def test_naive_preserves_name() -> None:
    fc = fit_naive(_SERIES, _HORIZON)
    assert fc.name == _SERIES.name


# ── fit_seasonal_naive ────────────────────────────────────────────────────────


def test_snv_length() -> None:
    fc = fit_seasonal_naive(_SERIES, _HORIZON)
    assert len(fc) == _HORIZON


def test_snv_period_index() -> None:
    fc = fit_seasonal_naive(_SERIES, _HORIZON)
    assert isinstance(fc.index, pd.PeriodIndex)
    assert fc.index[0] == _SERIES.index[-1] + 1


def test_snv_repeats_last_cycle() -> None:
    fc = fit_seasonal_naive(_SERIES, _HORIZON, period=4)
    for i in range(_HORIZON):
        expected = _SERIES.iloc[-4 + (i % 4)]
        assert abs(fc.iloc[i] - expected) < 1e-10, f"step {i}: {fc.iloc[i]} != {expected}"


def test_snv_all_finite() -> None:
    fc = fit_seasonal_naive(_SERIES, _HORIZON)
    assert np.isfinite(fc.values).all()


def test_snv_short_series_fallback() -> None:
    short = _SERIES.iloc[:3]  # 3 quarters < period=4 → falls back to naive
    fc = fit_seasonal_naive(short, _HORIZON)
    assert len(fc) == _HORIZON
    assert (fc == short.iloc[-1]).all()


# ── fit_ets ───────────────────────────────────────────────────────────────────


def test_ets_length() -> None:
    fc = fit_ets(_SERIES, _HORIZON)
    assert len(fc) == _HORIZON


def test_ets_period_index() -> None:
    fc = fit_ets(_SERIES, _HORIZON)
    assert isinstance(fc.index, pd.PeriodIndex)
    assert fc.index[0] == _SERIES.index[-1] + 1


def test_ets_all_finite() -> None:
    fc = fit_ets(_SERIES, _HORIZON)
    assert np.isfinite(fc.values).all()


def test_ets_returns_series() -> None:
    fc = fit_ets(_SERIES, _HORIZON)
    assert isinstance(fc, pd.Series)


def test_ets_short_series_fallback() -> None:
    short = _SERIES.iloc[:7]  # 7 < 2*4=8 → falls back to seasonal_naive
    fc = fit_ets(short, _HORIZON)
    assert len(fc) == _HORIZON
    assert np.isfinite(fc.values).all()


def test_ets_no_trend_no_seasonal() -> None:
    fc = fit_ets(_SERIES, _HORIZON, trend=None, seasonal=None)
    assert len(fc) == _HORIZON
    assert np.isfinite(fc.values).all()


def test_ets_robust_param_ignored() -> None:
    fc1 = fit_ets(_SERIES, _HORIZON, robust=True)
    fc2 = fit_ets(_SERIES, _HORIZON, robust=False)
    pd.testing.assert_series_equal(fc1, fc2)


# ── bottom_up_hierarchical ────────────────────────────────────────────────────


def test_buh_returns_dataframe(tourism_train: pd.DataFrame) -> None:
    result = bottom_up_hierarchical(tourism_train, "State", _HORIZON, fit_fn=fit_naive)
    assert isinstance(result, pd.DataFrame)


def test_buh_index_length(tourism_train: pd.DataFrame) -> None:
    result = bottom_up_hierarchical(tourism_train, "State", _HORIZON, fit_fn=fit_naive)
    assert len(result) == _HORIZON


def test_buh_period_index(tourism_train: pd.DataFrame) -> None:
    result = bottom_up_hierarchical(tourism_train, "State", _HORIZON, fit_fn=fit_naive)
    assert isinstance(result.index, pd.PeriodIndex)


def test_buh_state_columns(tourism_train: pd.DataFrame) -> None:
    result = bottom_up_hierarchical(tourism_train, "State", _HORIZON, fit_fn=fit_naive)
    expected = set(tourism_train["State"].unique())
    assert set(result.columns) == expected


def test_buh_all_finite(tourism_train: pd.DataFrame) -> None:
    result = bottom_up_hierarchical(tourism_train, "State", _HORIZON, fit_fn=fit_naive)
    assert np.isfinite(result.values).all()


def test_buh_algebraic_identity(tourism_train: pd.DataFrame) -> None:
    """Sum of bottom-up state forecasts must equal naive applied to the national aggregate."""
    from tsshowcase.data import aggregate_series

    national = aggregate_series(tourism_train)
    national_fc = fit_naive(national, _HORIZON)

    result = bottom_up_hierarchical(tourism_train, "State", _HORIZON, fit_fn=fit_naive)
    bu_total = result.sum(axis=1)

    pd.testing.assert_series_equal(
        bu_total.rename(None),
        national_fc.rename(None),
        check_names=False,
        rtol=1e-8,
    )


def test_buh_purpose_columns(tourism_train: pd.DataFrame) -> None:
    result = bottom_up_hierarchical(tourism_train, "Purpose", _HORIZON, fit_fn=fit_naive)
    expected = set(tourism_train["Purpose"].unique())
    assert set(result.columns) == expected
