# -*- coding: utf-8 -*-
"""Validate selected weekly data outputs against source files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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


if __name__ == "__main__":
    validate_output_files()
