# ts-forecasting skill

Baseline and bottom-up hierarchical forecasting for the Australian tourism dataset.

## When to use

Invoke this skill when asked to: fit a forecast, run baselines, apply ETS or naive
models, produce a bottom-up hierarchical forecast, run `models.py`, use
`baselines.py`, evaluate forecast accuracy, or work with the ts-showcase
forecasting layer generally.

## Module: `src/tsshowcase/models.py`

### Public API

```python
fit_naive(series, horizon, period=4)          → pd.Series  # future PeriodIndex
fit_seasonal_naive(series, horizon, period=4) → pd.Series
fit_ets(series, horizon, period=4,
        trend="add", seasonal="add",
        robust=False)                          → pd.Series
bottom_up_hierarchical(df_bottom, group_col,
                       horizon,
                       fit_fn=fit_ets)         → pd.DataFrame
```

All `fit_*` functions return a `pd.Series` with a future `PeriodIndex` of length
`horizon`. Signatures are uniform so any of them can be passed as `fit_fn`.

### `bottom_up_hierarchical`
- `df_bottom` — the training DataFrame (PeriodIndex, categorical cols + Trips float).
- `group_col` — column to aggregate to, e.g. `"State"` → 8 output columns.
- Returns a `pd.DataFrame` (index = future quarters, columns = group values).
- Internally: forecasts every leaf (Region × Purpose within the group), then
  `pd.concat(..., axis=1).sum(axis=1)` per group. Do NOT groupby on "Quarter"
  as a column — it is the DataFrame index.

## CLI: `scripts/baselines.py`

Standalone diagnostic script (no tsshowcase imports). Run from the project root:

```bash
.venv/bin/python .claude/skills/ts-forecasting/scripts/baselines.py \
    data/tourism.csv "New South Wales" Sydney Holiday --horizon 8
```

Prints a step-by-step forecast table (naive / seasonal_naive / ets) and a
summary line with last observed value and mean forecasts per model.

## Key gotchas

1. **ETS short-series guard** — `ExponentialSmoothing` with additive seasonal
   raises `ValueError` when `len(series) < 2 * period`. `fit_ets` falls back to
   `fit_seasonal_naive` automatically; `fit_seasonal_naive` falls back to
   `fit_naive` when `len(series) < period`.

2. **`robust` parameter** — accepted by `fit_ets` for API symmetry with
   `stl_decompose` but has no effect. statsmodels ETS has no robust mode.

3. **PeriodIndex in statsmodels 0.14** — `ExponentialSmoothing` accepts a
   `PeriodIndex` series directly. `model.forecast(horizon)` already returns the
   correct future `PeriodIndex`; no `.to_timestamp()` conversion is needed
   (unlike `plot_decomposition` in `eda.py`).

4. **Coding standards** — `from __future__ import annotations` required.
   Full type annotations. `ruff check` + `ruff format` before commit.

## Tests

```bash
.venv/bin/pytest tests/test_models.py -v
```

Synthetic 32-quarter fixture for unit tests; module-scoped real-data fixture
(skips if `data/tourism.csv` absent) for integration tests.
