"""Tests for tsshowcase.data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tsshowcase.data import (
    _parse_quarter,
    _process_raw,
    aggregate_series,
    get_series,
    list_keys,
    load_tourism,
    split_train_test,
)

_CACHE = Path("data/tourism.csv")


@pytest.fixture(scope="module")
def tourism() -> pd.DataFrame:
    if not _CACHE.exists():
        pytest.skip("tourism cache not present — run download_tourism() first")
    return load_tourism(_CACHE)


# ── _parse_quarter ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("March quarter 1998",     "1998Q1"),
    ("June quarter 2000",      "2000Q2"),
    ("September quarter 2010", "2010Q3"),
    ("December quarter 2017",  "2017Q4"),
])
def test_parse_quarter(raw: str, expected: str) -> None:
    assert _parse_quarter(raw) == pd.Period(expected, freq="Q")


# ── _process_raw ───────────────────────────────────────────────────────────────

def _make_raw() -> pd.DataFrame:
    """Minimal synthetic raw DataFrame mirroring the ABS SuperWEB2 layout.

    Structure: 2 states (A, B), 3 regions total, 2 quarters.
    State-total rows appear *after* their regions, as in the real file.
    Columns beyond index 5 are ignored by _process_raw.
    """
    rows = [
        # Q1 1998 — State A: 2 regions, then state-total row
        ["March quarter 1998", "Region A1", 10, 5, 3, 1, 0, 0, 0, 19],
        [None,                  "Region A2", 20, 8, 4, 2, 0, 0, 0, 34],
        [None,                  "State A",   30, 13, 7, 3, 0, 0, 0, 53],
        # Q1 1998 — State B: 1 region, then state-total row
        [None,                  "Region B1", 15, 6, 2, 0, 0, 0, 0, 23],
        [None,                  "State B",   15, 6, 2, 0, 0, 0, 0, 23],
        # Q2 1998 — State A
        ["June quarter 1998",   "Region A1", 12, 6, 3, 1, 0, 0, 0, 22],
        [None,                  "Region A2", 22, 9, 5, 2, 0, 0, 0, 38],
        [None,                  "State A",   34, 15, 8, 3, 0, 0, 0, 60],
        # Q2 1998 — State B
        [None,                  "Region B1", 16, 7, 3, 1, 0, 0, 0, 27],
        [None,                  "State B",   16, 7, 3, 1, 0, 0, 0, 27],
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_states(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch _STATES to match the synthetic raw DataFrame's state names."""
    monkeypatch.setattr("tsshowcase.data._STATES", frozenset({"State A", "State B"}))


def test_process_raw_shape(synthetic_states: None) -> None:
    # 3 regions × 4 purposes × 2 quarters = 24 rows, 4 columns
    assert _process_raw(_make_raw()).shape == (24, 4)


def test_process_raw_no_state_rows(synthetic_states: None) -> None:
    result = _process_raw(_make_raw())
    assert "State A" not in result["Region"].values
    assert "State B" not in result["Region"].values


def test_process_raw_state_assignment(synthetic_states: None) -> None:
    r = _process_raw(_make_raw()).reset_index()
    assert set(r[r["Region"].isin(["Region A1", "Region A2"])]["State"]) == {"State A"}
    assert set(r[r["Region"] == "Region B1"]["State"]) == {"State B"}


def test_process_raw_quarter_range(synthetic_states: None) -> None:
    result = _process_raw(_make_raw())
    assert result.index.min() == pd.Period("1998Q1", freq="Q")
    assert result.index.max() == pd.Period("1998Q2", freq="Q")


# ── load_tourism — shape & types ───────────────────────────────────────────────

def test_shape(tourism: pd.DataFrame) -> None:
    assert tourism.shape == (24_320, 4)  # 304 leaf series × 80 quarters


def test_quarter_range(tourism: pd.DataFrame) -> None:
    assert tourism.index.min() == pd.Period("1998Q1", freq="Q")
    assert tourism.index.max() == pd.Period("2017Q4", freq="Q")


def test_index_is_period(tourism: pd.DataFrame) -> None:
    assert isinstance(tourism.index, pd.PeriodIndex)
    assert tourism.index.freqstr == "Q-DEC"


def test_column_dtypes(tourism: pd.DataFrame) -> None:
    assert pd.api.types.is_float_dtype(tourism["Trips"])
    for col in ("State", "Region", "Purpose"):
        assert hasattr(tourism[col], "cat"), f"{col} should be Categorical"


def test_no_missing_trips(tourism: pd.DataFrame) -> None:
    assert tourism["Trips"].notna().all()


def test_trips_non_negative(tourism: pd.DataFrame) -> None:
    assert (tourism["Trips"] >= 0).all()


# ── load_tourism — hierarchy counts ───────────────────────────────────────────

def test_state_count(tourism: pd.DataFrame) -> None:
    assert tourism["State"].nunique() == 8


def test_region_count(tourism: pd.DataFrame) -> None:
    assert tourism["Region"].nunique() == 76


def test_purpose_values(tourism: pd.DataFrame) -> None:
    assert set(tourism["Purpose"].cat.categories) == {
        "Holiday",
        "Business",
        "Visiting friends and relatives",
        "Other reason",
    }


# ── list_keys ──────────────────────────────────────────────────────────────────

def test_list_keys_count(tourism: pd.DataFrame) -> None:
    keys = list_keys(tourism)
    assert len(keys) == 304


def test_list_keys_columns(tourism: pd.DataFrame) -> None:
    assert list(list_keys(tourism).columns) == ["State", "Region", "Purpose"]


def test_list_keys_no_duplicates(tourism: pd.DataFrame) -> None:
    keys = list_keys(tourism)
    assert not keys.duplicated().any()


# ── get_series ─────────────────────────────────────────────────────────────────

def test_get_series_length(tourism: pd.DataFrame) -> None:
    s = get_series(tourism, "New South Wales", "Sydney", "Holiday")
    assert len(s) == 80


def test_get_series_index_type(tourism: pd.DataFrame) -> None:
    s = get_series(tourism, "New South Wales", "Sydney", "Holiday")
    assert isinstance(s.index, pd.PeriodIndex)


def test_get_series_name(tourism: pd.DataFrame) -> None:
    s = get_series(tourism, "New South Wales", "Sydney", "Holiday")
    assert s.name == "New South Wales|Sydney|Holiday"


# ── aggregate_series ───────────────────────────────────────────────────────────

def test_aggregate_by_state_length(tourism: pd.DataFrame) -> None:
    assert len(aggregate_series(tourism, state="New South Wales")) == 80


def test_aggregate_national_equals_sum(tourism: pd.DataFrame) -> None:
    national = aggregate_series(tourism)
    manual = tourism.groupby(level="Quarter")["Trips"].sum()
    pd.testing.assert_series_equal(national, manual)


def test_aggregate_leaf_matches_get_series(tourism: pd.DataFrame) -> None:
    agg = aggregate_series(
        tourism, state="New South Wales", region="Sydney", purpose="Holiday"
    )
    direct = get_series(tourism, "New South Wales", "Sydney", "Holiday")
    pd.testing.assert_series_equal(agg, direct, check_names=False)


# ── split_train_test ───────────────────────────────────────────────────────────

def test_split_quarter_counts(tourism: pd.DataFrame) -> None:
    train, test = split_train_test(tourism)
    assert test.index.nunique() == 8
    assert train.index.nunique() == 72


def test_split_no_overlap(tourism: pd.DataFrame) -> None:
    train, test = split_train_test(tourism)
    assert set(train.index).isdisjoint(set(test.index))


def test_split_row_counts(tourism: pd.DataFrame) -> None:
    train, test = split_train_test(tourism)
    assert len(train) + len(test) == len(tourism)


def test_split_test_covers_end(tourism: pd.DataFrame) -> None:
    _, test = split_train_test(tourism)
    assert test.index.max() == pd.Period("2017Q4", freq="Q")


def test_split_custom_window(tourism: pd.DataFrame) -> None:
    _, test = split_train_test(tourism, test_quarters=4)
    assert test.index.nunique() == 4
