"""Download and load the Australian tourism dataset.

Source
------
Raw data: Tourism Research Australia (TRA) / Austrade, National Visitor Survey.
Delivered via tsibble R package (GPL-3), which processes an ABS SuperWEB2 export.
Raw URL: https://raw.githubusercontent.com/tidyverts/tsibble/main/data-raw/domestic-trips.csv

Dataset mirrors tsibble::tourism
  - 304 leaf series, 80 quarters (1998 Q1 – 2017 Q4)
  - Geographic hierarchy: National → 8 States → 76 Regions
  - Crossed with 4 travel purposes:
      Holiday | Business | Visiting friends and relatives | Other reason
  - Measure: Trips — overnight trips away from home, in thousands
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_RAW_URL = (
    "https://raw.githubusercontent.com/tidyverts/tsibble/"
    "main/data-raw/domestic-trips.csv"
)
_DEFAULT_PATH = Path("data/tourism.csv")

# State heading rows embedded in the Region column of the raw ABS export.
_STATES: frozenset[str] = frozenset(
    {
        "New South Wales",
        "Victoria",
        "Queensland",
        "South Australia",
        "Western Australia",
        "Tasmania",
        "Northern Territory",
        "ACT",  # raw file uses abbreviation, not full name
    }
)

# Raw column name → canonical Purpose label used in tsibble::tourism.
_PURPOSE_MAP: dict[str, str] = {
    "Holiday": "Holiday",
    "Visiting": "Visiting friends and relatives",
    "Business": "Business",
    "Other": "Other reason",
}

# Month name in "March quarter 1998" → ISO quarter number.
_QUARTER_MONTH: dict[str, str] = {
    "March": "Q1",
    "June": "Q2",
    "September": "Q3",
    "December": "Q4",
}


def _parse_quarter(q: str) -> pd.Period:
    """Convert 'March quarter 1998' → Period('1998Q1', freq='Q')."""
    parts = q.strip().split()
    return pd.Period(f"{parts[-1]}{_QUARTER_MONTH[parts[0]]}", freq="Q")


def _process_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the wide-format ABS DataFrame into tidy long format.

    Replicates the logic from tsibble/data-raw/read.R in Python.
    """
    # First 6 columns: Quarter, Region, Holiday, Visiting, Business, Other.
    df = df.iloc[:, :6].copy()
    df.columns = pd.Index(["Quarter", "Region", "Holiday", "Visiting", "Business", "Other"])

    df = df.dropna(how="all")  # strip blank footer rows

    # Quarter string only appears on the first row of each quarter block.
    df["Quarter"] = df["Quarter"].ffill()

    # Drop "Total" summary rows before extracting state labels.
    df = df[~df["Region"].astype(str).str.lower().str.startswith("total")]

    # State total rows appear *after* their regions in the raw file (e.g. all NSW
    # regions come first, then "New South Wales" as the block footer). Use backward
    # fill within each quarter group so each region row picks up the state that
    # follows it, not the one that precedes it.
    df["State"] = df["Region"].where(df["Region"].isin(_STATES))
    df["State"] = df.groupby("Quarter", sort=False)["State"].transform("bfill")

    # Drop the state total rows themselves — keep only leaf region rows.
    df = df[~df["Region"].isin(_STATES)]

    # Drop any rows whose Quarter didn't resolve to a valid "Month quarter YYYY" string
    # (e.g. stray "Total" summary rows in the quarter column).
    valid_months = set(_QUARTER_MONTH.keys())
    df = df[df["Quarter"].str.split().str[0].isin(valid_months)]

    df = df.dropna(subset=["Quarter", "Region", "State"])

    # Pivot Holiday / Visiting / Business / Other → long Purpose + Trips columns.
    df = df.melt(
        id_vars=["Quarter", "State", "Region"],
        value_vars=list(_PURPOSE_MAP),
        var_name="Purpose",
        value_name="Trips",
    )
    df["Purpose"] = df["Purpose"].map(_PURPOSE_MAP)

    # Parse "March quarter 1998" → quarterly Period.
    df["Quarter"] = df["Quarter"].map(_parse_quarter)

    for col in ("State", "Region", "Purpose"):
        df[col] = df[col].astype("category")
    df["Trips"] = pd.to_numeric(df["Trips"], errors="coerce").astype("float64")

    return df.set_index("Quarter").sort_index()


def download_tourism(dest: Path = _DEFAULT_PATH) -> Path:
    """Fetch, transform, and cache the tourism dataset as a long-format CSV.

    Downloads the raw ABS wide-format export from the tsibble GitHub repo
    (skipping 11 preamble rows), transforms it to match tsibble::tourism,
    and writes the result to *dest*. No-ops if *dest* already exists.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest

    # header=None: the first data row ("March quarter 1998", "Sydney", ...)
    # must not be consumed as column names. Columns are renamed inside _process_raw.
    raw = pd.read_csv(_RAW_URL, skiprows=11, header=None, encoding="utf-8")
    processed = _process_raw(raw)
    processed.reset_index().to_csv(dest, index=False)
    return dest


def load_tourism(path: Path = _DEFAULT_PATH) -> pd.DataFrame:
    """Load the tourism dataset with fully-typed columns.

    Downloads and transforms from source on first call.

    Returns a DataFrame indexed by Quarter (PeriodIndex, freq='Q') with columns:
      State    — pd.CategoricalDtype  (8 Australian states / territories)
      Region   — pd.CategoricalDtype  (76 tourism regions)
      Purpose  — pd.CategoricalDtype  (4 travel purposes)
      Trips    — float64              (overnight trips, thousands)
    """
    if not path.exists():
        download_tourism(path)

    df = pd.read_csv(path)
    df["Quarter"] = pd.PeriodIndex(df["Quarter"], freq="Q")
    for col in ("State", "Region", "Purpose"):
        df[col] = df[col].astype("category")
    df["Trips"] = df["Trips"].astype("float64")
    return df.set_index("Quarter").sort_index()


def list_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Return the unique (State, Region, Purpose) leaf combinations, sorted."""
    return (
        df.reset_index()[["State", "Region", "Purpose"]]
        .drop_duplicates()
        .sort_values(["State", "Region", "Purpose"])
        .reset_index(drop=True)
    )


def get_series(df: pd.DataFrame, state: str, region: str, purpose: str) -> pd.Series:
    """Extract a single leaf time series of Trips (thousands)."""
    mask = (
        (df["State"] == state)
        & (df["Region"] == region)
        & (df["Purpose"] == purpose)
    )
    return df.loc[mask, "Trips"].rename(f"{state}|{region}|{purpose}")


def aggregate_series(
    df: pd.DataFrame,
    *,
    state: str | None = None,
    region: str | None = None,
    purpose: str | None = None,
) -> pd.Series:
    """Sum Trips across any subset of dimensions, grouped by Quarter.

    Omit a keyword to aggregate across all values of that dimension.
    """
    filtered = df
    if state is not None:
        filtered = filtered[filtered["State"] == state]
    if region is not None:
        filtered = filtered[filtered["Region"] == region]
    if purpose is not None:
        filtered = filtered[filtered["Purpose"] == purpose]
    return filtered.groupby(level="Quarter")["Trips"].sum()


def split_train_test(
    df: pd.DataFrame,
    test_quarters: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into train / test by holding out the last *test_quarters* periods.

    Applied uniformly across all series so every leaf has exactly *test_quarters*
    test observations. The default of 8 gives a 2-year test window (2016 Q1 – 2017 Q4).
    """
    cutoff: pd.Period = df.index.unique().sort_values()[-test_quarters]
    return df[df.index < cutoff].copy(), df[df.index >= cutoff].copy()
