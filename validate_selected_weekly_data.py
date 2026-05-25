# -*- coding: utf-8 -*-
"""Validate the current selected weekly groundwater dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_SELECTED_OUT_DIR = WORKSPACE / "test" / "selected_weekly_data_15wells_current"
FIELD_ORDER = ["Date", "TASMAX", "TAS", "TASMIN", "Humidity", "Precipitation", "GWL"]
SUMMARY_NAME = "selected_wells_summary.csv"
COMBINED_NAME = "selected_wells_combined_long.csv"


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_weekly_frame(df: pd.DataFrame, *, label: str) -> None:
    assert_true(df.columns.tolist() == FIELD_ORDER, f"bad columns for {label}")
    assert_true(len(df) > 0, f"empty data for {label}")
    assert_true(df["Date"].is_monotonic_increasing, f"dates are not sorted for {label}")
    assert_true(not df["Date"].duplicated().any(), f"duplicate dates for {label}")
    assert_true(not df[FIELD_ORDER].isna().any().any(), f"null values found for {label}")

    for column in FIELD_ORDER:
        if column == "Date":
            continue
        assert_true(is_numeric_dtype(df[column]), f"non-numeric {column} values for {label}")


def validate_selected_well_output_files(
    out_dir: Path = DEFAULT_SELECTED_OUT_DIR,
    *,
    expected_well_count: int = 15,
    expected_per_type: int = 5,
) -> None:
    summary_path = out_dir / SUMMARY_NAME
    combined_path = out_dir / COMBINED_NAME
    assert_true(summary_path.exists(), f"missing summary: {summary_path}")
    assert_true(combined_path.exists(), f"missing combined data: {combined_path}")

    summary = pd.read_csv(summary_path)
    required_summary_columns = {
        "aquifer_type",
        "chinese_type",
        "well_id",
        "start_date",
        "end_date",
        "valid_weeks",
        "output_file",
    }
    missing = required_summary_columns - set(summary.columns)
    assert_true(not missing, f"missing summary columns: {sorted(missing)}")
    assert_true(len(summary) == expected_well_count, f"summary must contain exactly {expected_well_count} wells")
    assert_true(summary["well_id"].is_unique, "summary contains duplicate well_id values")
    assert_true(summary["output_file"].is_unique, "summary contains duplicate output files")
    expected_counts = {"f": expected_per_type, "k": expected_per_type, "p": expected_per_type}
    assert_true(
        summary["aquifer_type"].value_counts().to_dict() == expected_counts,
        f"expected {expected_per_type} wells per aquifer type",
    )

    total_rows = 0
    for _, row in summary.iterrows():
        output_path = out_dir / str(row["output_file"])
        well_id = str(row["well_id"])
        assert_true(output_path.exists(), f"missing output: {output_path}")

        df = pd.read_csv(output_path, parse_dates=["Date"])
        _validate_weekly_frame(df, label=well_id)
        assert_true(len(df) == int(row["valid_weeks"]), f"bad row count for {well_id}")
        assert_true(df["Date"].min() == pd.Timestamp(row["start_date"]), f"bad start date for {well_id}")
        assert_true(df["Date"].max() == pd.Timestamp(row["end_date"]), f"bad end date for {well_id}")
        total_rows += len(df)

    combined = pd.read_csv(combined_path, parse_dates=["Date"])
    required_combined_columns = ["aquifer_type", "chinese_type", "well_id", "label", *FIELD_ORDER]
    assert_true(combined.columns.tolist() == required_combined_columns, "bad combined long columns")
    assert_true(len(combined) == total_rows, "bad combined long row count")
    assert_true(not combined[required_combined_columns].isna().any().any(), "null values found in combined long data")

    grouped_sizes = combined.groupby("well_id").size()
    expected_sizes = summary.set_index("well_id")["valid_weeks"].astype(int)
    for well_id, expected_rows in expected_sizes.items():
        assert_true(int(grouped_sizes.loc[well_id]) == int(expected_rows), f"bad combined rows for {well_id}")

    duplicate_pairs = combined.duplicated(subset=["well_id", "Date"])
    assert_true(not duplicate_pairs.any(), "combined long contains duplicate well/date rows")
    print(f"{expected_well_count}-well validation passed: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SELECTED_OUT_DIR)
    parser.add_argument("--expected-well-count", type=int, default=15)
    parser.add_argument("--expected-per-type", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_selected_well_output_files(
        args.out_dir,
        expected_well_count=args.expected_well_count,
        expected_per_type=args.expected_per_type,
    )


if __name__ == "__main__":
    main()
