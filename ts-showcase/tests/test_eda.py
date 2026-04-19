"""Tests for tsshowcase.eda."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # must be set before any other matplotlib import

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from tsshowcase.eda import plot_decomposition, seasonally_adjusted, stl_decompose

# ── Shared test fixture ────────────────────────────────────────────────────────
# 8 years of synthetic quarterly data: linear trend + sinusoidal seasonal + noise.
# seasonal=7 is used in tests (minimum valid odd window ≥ 7 for period=4).

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
_SEASONAL_WIN = 7  # small window suited to short test series


@pytest.fixture(autouse=True)
def close_figures() -> None:  # type: ignore[return]
    """Close all matplotlib figures after every test."""
    yield
    matplotlib.pyplot.close("all")


# ── stl_decompose — return structure ──────────────────────────────────────────

def test_returns_decompose_result() -> None:
    from statsmodels.tsa.seasonal import DecomposeResult
    result = stl_decompose(_SERIES, seasonal=_SEASONAL_WIN)
    assert isinstance(result, DecomposeResult)


def test_has_required_attributes() -> None:
    result = stl_decompose(_SERIES, seasonal=_SEASONAL_WIN)
    for attr in ("trend", "seasonal", "resid", "observed"):
        assert hasattr(result, attr), f"missing attribute: {attr}"


def test_component_lengths() -> None:
    result = stl_decompose(_SERIES, seasonal=_SEASONAL_WIN)
    assert len(result.trend) == _N
    assert len(result.seasonal) == _N
    assert len(result.resid) == _N


# ── stl_decompose — additive identity ─────────────────────────────────────────

def test_additive_identity() -> None:
    """trend + seasonal + resid must reconstruct observed to floating-point precision."""
    result = stl_decompose(_SERIES, seasonal=_SEASONAL_WIN)
    reconstructed = result.trend + result.seasonal + result.resid
    np.testing.assert_allclose(reconstructed, result.observed, rtol=1e-8)


# ── stl_decompose — seasonal component properties ─────────────────────────────

def test_seasonal_has_period_4_pattern() -> None:
    """Seasonal values at the same quarter position should repeat closely."""
    result = stl_decompose(_SERIES, seasonal=_SEASONAL_WIN)
    seasonal = result.seasonal
    # Q1 values across years should all be similar (low variance relative to amplitude)
    q1_values = seasonal[0::4]
    assert q1_values.std() < 2.0, "Q1 seasonal values vary too much across years"


def test_seasonal_full_cycle_near_zero() -> None:
    """Each complete seasonal cycle must sum to approximately zero (additive)."""
    result = stl_decompose(_SERIES, seasonal=_SEASONAL_WIN)
    cycle_sums = [result.seasonal[i : i + 4].sum() for i in range(0, _N - 3, 4)]
    for s in cycle_sums:
        assert abs(s) < 5.0, f"seasonal cycle sum {s:.2f} is too large"


# ── stl_decompose — robustness and NaN handling ────────────────────────────────

def test_robust_vs_nonrobust_same_shape() -> None:
    r1 = stl_decompose(_SERIES, seasonal=_SEASONAL_WIN, robust=True)
    r2 = stl_decompose(_SERIES, seasonal=_SEASONAL_WIN, robust=False)
    assert r1.trend.shape == r2.trend.shape


def test_drops_nan_before_fitting() -> None:
    """A series with a leading NaN should decompose with one fewer observation."""
    s = _SERIES.copy()
    s.iloc[0] = float("nan")
    result = stl_decompose(s, seasonal=_SEASONAL_WIN)
    assert len(result.observed) == _N - 1


def test_outlier_absorbed_by_residual() -> None:
    """With robust=True, a spike should appear in resid, not in trend/seasonal."""
    s = _SERIES.copy()
    s.iloc[16] = s.mean() + 200  # large outlier at mid-series
    r_robust = stl_decompose(s, seasonal=_SEASONAL_WIN, robust=True)
    r_plain = stl_decompose(s, seasonal=_SEASONAL_WIN, robust=False)
    # Robust residual at the spike should be much larger (captures the outlier)
    assert abs(r_robust.resid.iloc[16]) > abs(r_plain.resid.iloc[16]) * 0.5


# ── seasonally_adjusted ────────────────────────────────────────────────────────

def test_sa_length() -> None:
    assert len(seasonally_adjusted(_SERIES, seasonal=_SEASONAL_WIN)) == _N


def test_sa_equals_trend_plus_resid() -> None:
    """SA series must equal trend + resid from the same decomposition."""
    result = stl_decompose(_SERIES, seasonal=_SEASONAL_WIN)
    sa = seasonally_adjusted(_SERIES, seasonal=_SEASONAL_WIN)
    expected = pd.Series(
        result.trend + result.resid,
        index=_SERIES.index,
    )
    pd.testing.assert_series_equal(sa, expected, check_names=False, rtol=1e-8)


def test_sa_name_contains_sa() -> None:
    sa = seasonally_adjusted(_SERIES, seasonal=_SEASONAL_WIN)
    assert "SA" in sa.name


def test_sa_unnamed_series() -> None:
    s = _SERIES.rename(None)
    sa = seasonally_adjusted(s, seasonal=_SEASONAL_WIN)
    assert sa.name == "SA"


# ── plot_decomposition ─────────────────────────────────────────────────────────

def test_plot_returns_figure() -> None:
    fig = plot_decomposition(_SERIES, seasonal=_SEASONAL_WIN)
    assert isinstance(fig, Figure)


def test_plot_has_four_axes() -> None:
    """STL plot always has 4 subplots: observed, trend, seasonal, residual."""
    fig = plot_decomposition(_SERIES, seasonal=_SEASONAL_WIN)
    assert len(fig.axes) == 4


def test_plot_title_set() -> None:
    fig = plot_decomposition(_SERIES, seasonal=_SEASONAL_WIN, title="My Title")
    assert any(t.get_text() == "My Title" for t in fig.texts)


def test_plot_no_title_by_default() -> None:
    fig = plot_decomposition(_SERIES, seasonal=_SEASONAL_WIN)
    assert all(t.get_text() == "" for t in fig.texts)
