"""Standalone baseline forecasting diagnostic for a single tourism leaf series.

Usage
-----
python baselines.py <csv_path> <state> <region> <purpose> [--horizon N]

Example
-------
python .claude/skills/ts-forecasting/scripts/baselines.py \\
    data/tourism.csv "New South Wales" Sydney Holiday --horizon 8

Intentionally self-contained: does not import from tsshowcase.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Inline model helpers (no tsshowcase dependency) ───────────────────────────

def _future_index(series: pd.Series, horizon: int) -> pd.PeriodIndex:
    last: pd.Period = series.index[-1]  # type: ignore[assignment]
    return pd.period_range(start=last + 1, periods=horizon, freq=last.freqstr)


def _fit_naive(series: pd.Series, horizon: int, period: int = 4) -> pd.Series:
    return pd.Series(
        np.full(horizon, series.iloc[-1]),
        index=_future_index(series, horizon),
    )


def _fit_seasonal_naive(
    series: pd.Series, horizon: int, period: int = 4
) -> pd.Series:
    if len(series) < period:
        return _fit_naive(series, horizon)
    last_cycle = series.iloc[-period:].values
    values = np.tile(last_cycle, (horizon // period) + 1)[:horizon]
    return pd.Series(values, index=_future_index(series, horizon))


def _fit_ets(series: pd.Series, horizon: int, period: int = 4) -> pd.Series:
    if len(series) < 2 * period:
        return _fit_seasonal_naive(series, horizon, period=period)
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    model = ExponentialSmoothing(
        series, trend="add", seasonal="add", seasonal_periods=period
    ).fit(optimized=True)
    return model.forecast(horizon)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_series(
    csv_path: Path, state: str, region: str, purpose: str
) -> pd.Series:
    df = pd.read_csv(csv_path)
    df["Quarter"] = pd.PeriodIndex(df["Quarter"], freq="Q")
    df = df.set_index("Quarter").sort_index()
    mask = (
        (df["State"] == state)
        & (df["Region"] == region)
        & (df["Purpose"] == purpose)
    )
    series = df.loc[mask, "Trips"]
    if series.empty:
        sys.exit(
            f"No data found for: State={state!r}, Region={region!r}, "
            f"Purpose={purpose!r}\nCheck spelling — values are case-sensitive."
        )
    return series


# ── Output formatting ─────────────────────────────────────────────────────────

def _print_table(
    series: pd.Series,
    state: str,
    region: str,
    purpose: str,
    horizon: int,
) -> None:
    fc_naive = _fit_naive(series, horizon)
    fc_snv = _fit_seasonal_naive(series, horizon)
    fc_ets = _fit_ets(series, horizon)

    print(f"\nSeries : {state} | {region} | {purpose}")
    print(
        f"Periods: {series.index.min()} – {series.index.max()}"
        f"  ({len(series)} observations)"
    )
    print()

    col_w = 14
    header = (
        f"{'step':>4} | {'naive':>{col_w}} | {'seasonal_naive':>{col_w}} | {'ets':>{col_w}}"
    )
    print(header)
    print("-" * len(header))
    for i in range(horizon):
        print(
            f"{i + 1:>4} | {fc_naive.iloc[i]:>{col_w}.2f} |"
            f" {fc_snv.iloc[i]:>{col_w}.2f} |"
            f" {fc_ets.iloc[i]:>{col_w}.2f}"
        )

    print()
    print(f"Last observed : {series.iloc[-1]:.2f}  ({series.index[-1]})")
    print(
        f"Mean forecast — "
        f"naive: {fc_naive.mean():.2f}  "
        f"seasonal_naive: {fc_snv.mean():.2f}  "
        f"ets: {fc_ets.mean():.2f}"
    )
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline forecasts for a single Australian tourism leaf series."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the processed tourism CSV.")
    parser.add_argument("state", help="State name (e.g. 'New South Wales').")
    parser.add_argument("region", help="Region name (e.g. 'Sydney').")
    parser.add_argument("purpose", help="Travel purpose (e.g. 'Holiday').")
    parser.add_argument(
        "--horizon", type=int, default=8, help="Forecast horizon in quarters (default: 8)."
    )
    args = parser.parse_args()

    series = _load_series(args.csv_path, args.state, args.region, args.purpose)
    _print_table(series, args.state, args.region, args.purpose, args.horizon)


if __name__ == "__main__":
    main()
