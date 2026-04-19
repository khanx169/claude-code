# ts-showcase

Australian tourism time-series EDA and hierarchical forecast showcase.
Data: tsibbledata `tourism` — 425 quarterly series (1998 Q1 – 2017 Q4).
Structure: geographic hierarchy (National → 8 States → 76 Regions) crossed with
4 travel purposes (Holiday, Business, Visiting Friends/Relatives, Other).
Measure: overnight trips away from home, in thousands.

## What this project does
1. **EDA** — seasonal decomposition and trend visualisation per series.
2. **Forecast** — bottom-up hierarchical ETS forecasts aggregated to any level.

## Coding standards
- Every function and method must carry full type annotations.
- All code must pass `ruff check` and `ruff format` before commit.
- No bare `# noqa` suppressions — add an inline justification if one is needed.

## Where things live
- `src/tsshowcase/` — importable library: `data`, `eda`, `models`, `evaluate`
- `notebooks/demo.ipynb` — primary live-presentation artefact
- `tests/` — pytest tests; run with `pytest` from the project root
