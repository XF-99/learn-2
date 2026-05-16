# -*- coding: utf-8 -*-
"""Validate selected weekly data outputs against source files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from prepare_selected_weekly_data import (
    DEFAULT_GW_DIR,
    DEFAULT_MET_DIR,
    DEFAULT_OUT_DIR,
    FIELD_ORDER,
    METEO_SPECS,
    SELECTED_WELLS,
    aggregate_for_observation_dates,
    read_groundwater,
    read_meteo_daily,
)

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_NINE_WELL_OUT_DIR = WORKSPACE / "selected_weekly_data_9wells_common"
NINE_WELL_SUMMARY = "nine_wells_summary.csv"
NINE_WELL_COMBINED = "nine_wells_combined_long.csv"


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_output_files(out_dir: Path = DEFAULT_OUT_DIR) -> None:
    summary_path = out_dir / "selected_wells_summary.csv"
    assert_true(summary_path.exists(), f"missing summary: {summary_path}")

    summary = pd.read_csv(summary_path)
    assert_true(len(summary) == 3, "summary must contain exactly 3 wells")

    for well in SELECTED_WELLS:
        output_path = out_dir / well.output_name
        assert_true(output_path.exists(), f"missing output: {output_path}")
        df = pd.read_csv(output_path, parse_dates=["Date"])
        assert_true(df.columns.tolist() == FIELD_ORDER, f"bad columns for {well.well_id}")
        assert_true(df["Date"].is_monotonic_increasing, f"dates are not sorted for {well.well_id}")
        assert_true(not df["Date"].duplicated().any(), f"duplicate dates for {well.well_id}")
        assert_true(not df[FIELD_ORDER].isna().any().any(), f"null values found for {well.well_id}")

        gw = read_groundwater(DEFAULT_GW_DIR, well.well_id)
        assert_true(df["Date"].min() >= gw["Date"].min(), f"start before GW source for {well.well_id}")
        assert_true(df["Date"].max() <= gw["Date"].max(), f"end after GW source for {well.well_id}")

        sampled = df.iloc[[0, len(df) // 2, len(df) - 1]].copy()
        for spec in METEO_SPECS:
            daily = read_meteo_daily(DEFAULT_MET_DIR, well.well_id, spec)
            recomputed = aggregate_for_observation_dates(sampled["Date"], daily, spec.column, spec.agg)
            actual = sampled[spec.output_column].reset_index(drop=True).round(10)
            expected = recomputed.reset_index(drop=True).astype(float).round(10)
            assert_true(actual.equals(expected), f"aggregation mismatch for {well.well_id} {spec.output_column}")

        source_gwl = gw.set_index("Date")["GWL"].reindex(df["Date"]).reset_index(drop=True)
        actual_gwl = df["GWL"].reset_index(drop=True)
        assert_true(actual_gwl.round(10).equals(source_gwl.round(10)), f"GWL mismatch for {well.well_id}")

        print(
            f"{well.well_id}: rows={len(df)}, "
            f"range={df['Date'].min().date()} to {df['Date'].max().date()}"
        )

    print("validation passed")


def _validate_weekly_frame(
    df: pd.DataFrame,
    *,
    label: str,
    expected_rows: int,
    expected_start: pd.Timestamp,
    expected_end: pd.Timestamp,
) -> None:
    assert_true(df.columns.tolist() == FIELD_ORDER, f"bad columns for {label}")
    assert_true(len(df) == expected_rows, f"bad row count for {label}: {len(df)} != {expected_rows}")
    assert_true(df["Date"].is_monotonic_increasing, f"dates are not sorted for {label}")
    assert_true(not df["Date"].duplicated().any(), f"duplicate dates for {label}")
    assert_true(not df[FIELD_ORDER].isna().any().any(), f"null values found for {label}")
    assert_true(df["Date"].min() == expected_start, f"bad start date for {label}")
    assert_true(df["Date"].max() == expected_end, f"bad end date for {label}")

    for column in FIELD_ORDER:
        if column == "Date":
            continue
        assert_true(is_numeric_dtype(df[column]), f"non-numeric {column} values for {label}")


def validate_nine_well_output_files(out_dir: Path = DEFAULT_NINE_WELL_OUT_DIR) -> None:
    summary_path = out_dir / NINE_WELL_SUMMARY
    combined_path = out_dir / NINE_WELL_COMBINED
    assert_true(summary_path.exists(), f"missing summary: {summary_path}")
    assert_true(combined_path.exists(), f"missing combined data: {combined_path}")

    summary = pd.read_csv(summary_path)
    required_summary_columns = {
        "aquifer_type",
        "chinese_type",
        "well_id",
        "common_start_date",
        "common_end_date",
        "common_weeks",
        "output_file",
    }
    missing = required_summary_columns - set(summary.columns)
    assert_true(not missing, f"missing summary columns: {sorted(missing)}")
    assert_true(len(summary) == 9, "summary must contain exactly 9 wells")
    assert_true(summary["well_id"].is_unique, "summary contains duplicate well_id values")
    assert_true(summary["output_file"].is_unique, "summary contains duplicate output files")
    assert_true(summary["aquifer_type"].value_counts().to_dict() == {"f": 3, "k": 3, "p": 3}, "expected 3 wells per aquifer type")

    expected_rows = int(summary["common_weeks"].iloc[0])
    expected_start = pd.Timestamp(summary["common_start_date"].iloc[0])
    expected_end = pd.Timestamp(summary["common_end_date"].iloc[0])
    assert_true((summary["common_weeks"] == expected_rows).all(), "common_weeks must match across all 9 wells")
    assert_true((pd.to_datetime(summary["common_start_date"]) == expected_start).all(), "common_start_date must match across all 9 wells")
    assert_true((pd.to_datetime(summary["common_end_date"]) == expected_end).all(), "common_end_date must match across all 9 wells")

    reference_dates = None
    for _, row in summary.iterrows():
        output_path = out_dir / str(row["output_file"])
        well_id = str(row["well_id"])
        assert_true(output_path.exists(), f"missing output: {output_path}")

        df = pd.read_csv(output_path, parse_dates=["Date"])
        _validate_weekly_frame(
            df,
            label=well_id,
            expected_rows=expected_rows,
            expected_start=expected_start,
            expected_end=expected_end,
        )

        dates = df["Date"].reset_index(drop=True)
        if reference_dates is None:
            reference_dates = dates
        else:
            assert_true(dates.equals(reference_dates), f"date grid differs for {well_id}")

    combined = pd.read_csv(combined_path, parse_dates=["Date"])
    required_combined_columns = ["aquifer_type", "chinese_type", "well_id", *FIELD_ORDER]
    assert_true(combined.columns.tolist() == required_combined_columns, "bad combined long columns")
    assert_true(len(combined) == expected_rows * len(summary), "bad combined long row count")
    assert_true(not combined[required_combined_columns].isna().any().any(), "null values found in combined long data")

    grouped_sizes = combined.groupby("well_id").size()
    assert_true((grouped_sizes == expected_rows).all(), "combined long row count must match for every well")
    duplicate_pairs = combined.duplicated(subset=["well_id", "Date"])
    assert_true(not duplicate_pairs.any(), "combined long contains duplicate well/date rows")

    for well_id, df in combined.groupby("well_id"):
        dates = df.sort_values("Date")["Date"].reset_index(drop=True)
        assert_true(reference_dates is not None and dates.equals(reference_dates), f"combined date grid differs for {well_id}")

    print("9-well validation passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["nine", "legacy", "both"], default="nine")
    parser.add_argument("--legacy-out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--nine-out-dir", type=Path, default=DEFAULT_NINE_WELL_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset in {"legacy", "both"}:
        validate_output_files(args.legacy_out_dir)
    if args.dataset in {"nine", "both"}:
        validate_nine_well_output_files(args.nine_out_dir)


if __name__ == "__main__":
    main()
