"""Baseline forecasting models and bottom-up hierarchical aggregation.

Reference: Hyndman & Athanasopoulos, FPP3 §11.2 — Bottom-up forecasting.
  https://otexts.com/fpp3/bottom-up.html

Bottom-up approach: forecast every leaf series independently using a chosen
model, then sum coherently to produce aggregate-level forecasts.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def _future_index(series: pd.Series, horizon: int) -> pd.PeriodIndex:
    """Return the next *horizon* periods immediately after *series* ends."""
    last: pd.Period = series.index[-1]  # type: ignore[assignment]
    return pd.period_range(start=last + 1, periods=horizon, freq=last.freqstr)


def fit_naive(series: pd.Series, horizon: int, period: int = 4) -> pd.Series:
    """Forecast by repeating the last observed value for *horizon* steps.

    *period* is accepted for a uniform call signature across all fit_* functions
    (so any of them can be passed as *fit_fn* to bottom_up_hierarchical)
    but has no effect on the naive forecast.
    """
    return pd.Series(
        np.full(horizon, series.iloc[-1]),
        index=_future_index(series, horizon),
        name=series.name,
    )


def fit_seasonal_naive(
    series: pd.Series, horizon: int, period: int = 4
) -> pd.Series:
    """Forecast by repeating the last complete seasonal cycle.

    Falls back to *fit_naive* when ``len(series) < period``.
    """
    if len(series) < period:
        return fit_naive(series, horizon)
    last_cycle = series.iloc[-period:].values
    values = np.tile(last_cycle, (horizon // period) + 1)[:horizon]
    return pd.Series(
        values,
        index=_future_index(series, horizon),
        name=series.name,
    )


def fit_ets(
    series: pd.Series,
    horizon: int,
    period: int = 4,
    trend: str | None = "add",
    seasonal: str | None = "add",
    robust: bool = False,
) -> pd.Series:
    """Fit a Holt-Winters ETS model and return *horizon*-step-ahead forecasts.

    Falls back to *fit_seasonal_naive* when the series is shorter than
    ``2 * period`` observations — the minimum statsmodels requires to
    initialise additive seasonal smoothing.

    Note: *robust* is accepted for API symmetry with *stl_decompose* but has
    no effect. statsmodels ExponentialSmoothing has no robust fitting mode.
    """
    _ = robust  # documented no-op — statsmodels ETS has no equivalent
    if len(series) < 2 * period:
        return fit_seasonal_naive(series, horizon, period=period)
    model = ExponentialSmoothing(
        series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=period,
    ).fit(optimized=True)
    # statsmodels 0.14 preserves PeriodIndex through forecast(); no conversion needed.
    return model.forecast(horizon).rename(series.name)


def bottom_up_hierarchical(
    df_bottom: pd.DataFrame,
    group_col: str,
    horizon: int,
    fit_fn: Callable[[pd.Series, int], pd.Series] = fit_ets,
) -> pd.DataFrame:
    """Produce coherent aggregate forecasts using the bottom-up method.

    Forecasts every leaf series in *df_bottom* independently with *fit_fn*,
    then sums within each value of *group_col* to form higher-level coherent
    forecasts. All output columns sum to the national-total forecast (per FPP3
    §11.2: bottom-up forecasts are coherent by construction).

    Parameters
    ----------
    df_bottom:
        Training DataFrame with a PeriodIndex named "Quarter" and categorical
        columns plus a float ``Trips`` column. All rows are treated as leaf
        observations — no pre-aggregation is applied.
    group_col:
        Categorical column to aggregate to.
        ``"State"``   → 8 output columns, one per Australian state.
        ``"Purpose"`` → 4 output columns, one per travel purpose.
    horizon:
        Number of future periods to forecast.
    fit_fn:
        Callable with signature ``(series: pd.Series, horizon: int) → pd.Series``.
        Defaults to *fit_ets*. Pass *fit_naive* or *fit_seasonal_naive* for
        speed or when short series are expected.

    Returns
    -------
    pd.DataFrame
        Index: future PeriodIndex of length *horizon*, named "Quarter".
        Columns: unique observed values of *group_col* in *df_bottom*.
    """
    dim_cols: list[str] = [c for c in df_bottom.columns if c != "Trips"]
    leaf_cols: list[str] = [c for c in dim_cols if c != group_col]

    result: dict[str, pd.Series] = {}
    for g in df_bottom[group_col].unique():
        group_df = df_bottom[df_bottom[group_col] == g]
        leaf_keys = group_df.reset_index()[leaf_cols].drop_duplicates()

        fc_list: list[pd.Series] = []
        for _, key_row in leaf_keys.iterrows():
            # Filter progressively to the single leaf series.
            leaf_df = group_df
            for col in leaf_cols:
                leaf_df = leaf_df[leaf_df[col] == key_row[col]]
            leaf_series = leaf_df["Trips"]
            if len(leaf_series) > 0:
                fc_list.append(fit_fn(leaf_series, horizon))

        if fc_list:
            result[str(g)] = pd.concat(fc_list, axis=1).sum(axis=1)

    df_result = pd.DataFrame(result)
    df_result.index.name = "Quarter"
    return df_result
