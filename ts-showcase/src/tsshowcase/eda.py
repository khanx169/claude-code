"""EDA: STL seasonal decomposition and visualisation for the tourism dataset.

Reference: Hyndman & Athanasopoulos, FPP3 §3.6 — STL Decomposition.
  https://otexts.com/fpp3/stl.html

STL (Seasonal and Trend decomposition using Loess) decomposes additively:
    observed = trend + seasonal + residual

For multiplicative seasonality, log-transform the series before calling
and exponentiate the components afterwards.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL, DecomposeResult


def stl_decompose(
    series: pd.Series,
    period: int = 4,
    seasonal: int = 13,
    robust: bool = True,
) -> DecomposeResult:
    """Decompose *series* into trend-cycle, seasonal, and residual via STL.

    Parameters
    ----------
    series:
        Univariate time series (PeriodIndex or DatetimeIndex). Leading/trailing
        NaN values are dropped before fitting.
    period:
        Observations per seasonal cycle. 4 for quarterly data.
    seasonal:
        Seasonal LOESS smoother window length (odd integer ≥ 7). Larger values
        produce a more stable seasonal shape that changes slowly across years.
        FPP3 recommends 13 for quarterly data as a robust default; use
        ``'periodic'`` to force an identical seasonal pattern each year.
    robust:
        If True (recommended), uses iteratively re-weighted LOESS so that
        outliers accumulate in the residual rather than distorting the trend
        or seasonal components.

    Returns
    -------
    DecomposeResult
        Attributes: ``.trend``, ``.seasonal``, ``.resid``, ``.observed``.
        Satisfies: ``observed ≈ trend + seasonal + resid`` element-wise.
    """
    return STL(
        series.dropna(),
        period=period,
        seasonal=seasonal,
        robust=robust,
    ).fit()


def seasonally_adjusted(
    series: pd.Series,
    period: int = 4,
    seasonal: int = 13,
    robust: bool = True,
) -> pd.Series:
    """Return the seasonally adjusted series (observed − seasonal component).

    The result removes only the periodic seasonal pattern, leaving the
    trend-cycle and irregular noise intact.
    """
    result = stl_decompose(series, period=period, seasonal=seasonal, robust=robust)
    sa = series.dropna() - result.seasonal
    name = f"{series.name} (SA)" if series.name else "SA"
    return sa.rename(name)


def plot_decomposition(
    series: pd.Series,
    period: int = 4,
    seasonal: int = 13,
    robust: bool = True,
    title: str = "",
) -> plt.Figure:
    """Plot observed, trend, seasonal, and residual from an STL decomposition.

    Returns the matplotlib Figure for further customisation or saving.
    """
    # statsmodels' DecomposeResult.plot() requires a DatetimeIndex-compatible
    # series; convert PeriodIndex → DatetimeIndex before fitting for the plot.
    s = series.copy()
    if isinstance(s.index, pd.PeriodIndex):
        s.index = s.index.to_timestamp()

    from statsmodels.tsa.seasonal import STL as _STL

    result = _STL(s.dropna(), period=period, seasonal=seasonal, robust=robust).fit()
    fig = result.plot()
    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig
