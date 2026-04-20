---
name: ts-showcase project standards
description: Coding standards, tooling, and architecture conventions for the ts-showcase project
type: project
---

Project is a Python 3.9+ library for Australian tourism time-series EDA and hierarchical forecasting.

Standards (from CLAUDE.md + pyproject.toml):
- Every function and method must carry full type annotations
- All code must pass `ruff check` and `ruff format` (line-length=88, select=["E","W","F","I"])
- No bare `# noqa` suppressions — add inline justification
- `from __future__ import annotations` required (target-version = py39 in pyproject.toml)
- Tests run with `pytest` from project root

**Why:** Presented as a showcase; code quality is a first-class concern.
**How to apply:** Flag any annotation gap, ruff violation, or undocumented suppression as at least Minor severity.

Key architecture:
- `src/tsshowcase/data.py` — download, process, load, filter tourism CSV
- `src/tsshowcase/eda.py` — STL decomposition and plotting
- `src/tsshowcase/models.py` — fit_naive, fit_seasonal_naive, fit_ets, bottom_up_hierarchical
- `.claude/skills/ts-forecasting/scripts/baselines.py` — self-contained CLI script, intentionally duplicates model helpers (no tsshowcase import)
- Tests require `data/tourism.csv` cache; integration tests skip if absent

Linter: `.venv/bin/ruff check src/ tests/ .claude/` and `.venv/bin/ruff format --check src/ tests/ .claude/`
