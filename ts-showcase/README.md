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

## Data

`tsibbledata::tourism` — 425 quarterly series, 1998 Q1 – 2017 Q4.  
Geographic hierarchy: National → 8 States → 76 Regions, crossed with 4 travel purposes.  
Measure: overnight trips (thousands).
