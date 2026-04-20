Run a seasonal-naïve and ETS forecast for every series in the requested region.

The region name is: $ARGUMENTS

## Steps

1. Look up the state and available purposes for this region:

```bash
.venv/bin/python -c "
from pathlib import Path
from tsshowcase.data import load_tourism, list_keys
keys = list_keys(load_tourism(Path('data/tourism.csv')))
matches = keys[keys['Region'] == '$ARGUMENTS'][['State', 'Region', 'Purpose']]
if matches.empty:
    available = sorted(keys['Region'].unique())
    print(f'Region not found. Available regions ({len(available)}):')
    print(', '.join(available))
else:
    print(matches.to_string(index=False))
"
```

2. For each (State, Purpose) row returned, run:

```bash
.venv/bin/python .claude/skills/ts-forecasting/scripts/baselines.py \
    data/tourism.csv "<State>" "$ARGUMENTS" "<Purpose>" --horizon 8
```

3. Print all tables in sequence, one per purpose, with a clear header separating them.
   If the region was not found in step 1, stop after printing the available-regions list.
