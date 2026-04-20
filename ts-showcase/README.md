# ts-showcase

EDA and hierarchical forecasting showcase using the Australian tourism dataset.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
# Run tests
pytest

# Launch notebook
jupyter notebook notebooks/demo.ipynb

# Forecast a region via custom command (inside Claude Code)
/forecast "Sydney"
```

## Tour for trainees

**CLAUDE.md** is the project's persistent context file — Claude Code reads it automatically at the start of every session. It records what the project does, coding standards (type annotations, ruff), and where things live, so Claude never needs to re-discover conventions from scratch.

**Custom slash commands** (`.claude/commands/`) let you package a repeatable workflow as a single prompt template. `/forecast <region>` is defined in `forecast.md`: when you type it inside Claude Code, Claude looks up the region's state and purposes, runs `baselines.py` for each series, and prints a forecast table — no manual copy-pasting required.

**Custom subagents** (`.claude/agents/`) are specialised Claude instances with their own system prompt, tool restrictions, and persistent memory. This project ships two: `code-quality-reviewer` runs ruff, checks annotations, and scores test quality; `model-correctness-reviewer` audits forecasting implementations against FPP3. Invoke them with `subagent_type: "code-quality-reviewer"` inside the Agent tool.

**Skills** (`.claude/skills/`) are reference documents that Claude loads when working in a domain. The `ts-forecasting` skill (`SKILL.md`) documents the public API, key gotchas (ETS short-series guard, PeriodIndex handling), and the CLI interface of the standalone `baselines.py` diagnostic script — so Claude applies the right patterns without re-reading the source every time.

**Agent memory** (`.claude/agent-memory/`) gives each subagent a file-based long-term memory. The `code-quality-reviewer` writes what it learns about this codebase — recurring issues, tool invocations, project-specific standards — so each subsequent review starts with institutional knowledge already loaded.

## Data

`tsibble::tourism` — 304 leaf series × 80 quarters (1998 Q1 – 2017 Q4).  
Geographic hierarchy: National → 8 States → 76 Regions, crossed with 4 travel purposes.  
Measure: overnight trips away from home, in thousands.

**Source:** Tourism Research Australia (TRA) / Austrade, National Visitor Survey.  
Raw data distributed via the [`tsibble`](https://github.com/tidyverts/tsibble) R package (GPL-3).

**License:** No explicit open-data licence has been published by TRA/Austrade for this
dataset. The raw file carries a platform copyright notice
`© Space-Time Research 2013` (the ABS SuperWEB2 delivery tool). The `tsibble` R package
that packages and redistributes the data is GPL-3. This dataset is widely used in
academic and teaching contexts (see Hyndman & Athanasopoulos, *Forecasting: Principles
and Practice*, 3rd ed., 2021). If you intend to use the data commercially, contact
Tourism Research Australia directly: tourism.research@tra.gov.au
