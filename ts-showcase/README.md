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
