# -*- coding: utf-8 -*-
"""Prepare 9 selected wells with a shared weekly date range.

The selected wells keep the previous three wells and add two more wells per
aquifer class. Meteorological variables are aggregated over the 7 days ending
on each groundwater observation date.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_GW_DIR = Path(r"C:\Users\xf-99\Downloads\GWData (1)\GWData")
DEFAULT_OUT_DIR = WORKSPACE / "selected_weekly_data_9wells_common"

FIELD_ORDER = ["Date", "TASMAX", "TAS", "TASMIN", "Humidity", "Precipitation", "GWL"]


@dataclass(frozen=True)
class WellSelection:
    aquifer_type: str
    chinese_type: str
    well_id: str
    name: str

    @property
    def output_name(self) -> str:
        return f"{self.chinese_type}_{self.well_id}_每周数据.csv"


@dataclass(frozen=True)
class MeteoSpec:
    source_column: str
    output_column: str
    agg: str


SELECTED_WELLS = [
    WellSelection("f", "裂隙水", "HE_6253", "NETRA"),
    WellSelection("f", "裂隙水", "HE_12117", "HE_12117"),
    WellSelection("f", "裂隙水", "HE_7824", "HE_7824"),
    WellSelection("k", "岩溶水", "BY_15120", "IHRLERSTEIN TIEF K1"),
    WellSelection("k", "岩溶水", "BY_11119", "BY_11119"),
    WellSelection("k", "岩溶水", "BY_7126", "BY_7126"),
    WellSelection("p", "孔隙水", "SN_46460564", "Walda"),
    WellSelection("p", "孔隙水", "SN_49430964", "SN_49430964"),
    WellSelection("p", "孔隙水", "SN_49484004", "SN_49484004"),
]


METEO_SPECS = [
    MeteoSpec("TASMAX_C", "TASMAX", "max"),
    MeteoSpec("TAS_C", "TAS", "mean"),
    MeteoSpec("TASMIN_C", "TASMIN", "min"),
    MeteoSpec("Humidity_Percent", "Humidity", "mean"),
    MeteoSpec("Precipitation_mm", "Precipitation", "sum"),
]


def default_met_dir() -> Path:
    desktop = Path.home() / "Desktop"
    root = next(path for path in desktop.iterdir() if path.name == "气象数据")
    return next(path for path in root.iterdir() if path.name == "气象数据")


def read_groundwater(gw_dir: Path, well_id: str) -> pd.DataFrame:
    path = gw_dir / f"{well_id}_GW-Data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing groundwater file: {path}")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["GWL"] = pd.to_numeric(df["GWL"], errors="coerce")
    df = df.dropna(subset=["Date", "GWL"]).sort_values("Date")
    return df[["Date", "GWL"]].drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)


def find_meteo_file(met_dir: Path, well_id: str, source_column: str) -> Path:
    matches = []
    for path in met_dir.rglob(f"{well_id}_*.csv"):
        try:
            columns = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        if source_column in columns:
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {source_column} file for {well_id}, found {len(matches)}")
    return matches[0]


def read_meteo_daily(met_dir: Path, well_id: str, spec: MeteoSpec) -> pd.DataFrame:
    path = find_meteo_file(met_dir, well_id, spec.source_column)
    df = pd.read_csv(path, usecols=["Date", spec.source_column])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[spec.source_column] = pd.to_numeric(df[spec.source_column], errors="coerce")
    df = df.dropna(subset=["Date", spec.source_column]).sort_values("Date")
    return df[["Date", spec.source_column]].drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)


def aggregate_for_observation_dates(
    observation_dates: pd.Series,
    daily: pd.DataFrame,
    value_column: str,
    agg: str,
) -> pd.Series:
    daily_series = daily.set_index("Date")[value_column].sort_index()
    reducer: Callable[[pd.Series], float] = getattr(pd.Series, agg)
    values = []

    for obs_date in observation_dates:
        start = obs_date - pd.Timedelta(days=6)
        window = daily_series.loc[start:obs_date]
        values.append(float(reducer(window)) if len(window) == 7 else pd.NA)

    return pd.Series(values, index=observation_dates.index)


def prepare_well(well: WellSelection, gw_dir: Path, met_dir: Path) -> pd.DataFrame:
    result = read_groundwater(gw_dir, well.well_id)

    for spec in METEO_SPECS:
        daily = read_meteo_daily(met_dir, well.well_id, spec)
        result[spec.output_column] = aggregate_for_observation_dates(
            observation_dates=result["Date"],
            daily=daily,
            value_column=spec.source_column,
            agg=spec.agg,
        )

    return result[FIELD_ORDER].dropna(subset=FIELD_ORDER).sort_values("Date").reset_index(drop=True)


def crop_to_common_dates(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    common_dates: set[pd.Timestamp] | None = None
    for df in frames.values():
        dates = set(pd.to_datetime(df["Date"]))
        common_dates = dates if common_dates is None else common_dates & dates
    if not common_dates:
        raise ValueError("No shared weekly dates across the selected wells.")

    ordered_dates = sorted(common_dates)
    cropped = {}
    for well_id, df in frames.items():
        out = df[df["Date"].isin(ordered_dates)].copy()
        out = out.sort_values("Date").reset_index(drop=True)
        cropped[well_id] = out
    return cropped


def build_summary(well: WellSelection, df: pd.DataFrame, source_df: pd.DataFrame) -> dict[str, object]:
    return {
        "aquifer_type": well.aquifer_type,
        "chinese_type": well.chinese_type,
        "well_id": well.well_id,
        "name": well.name,
        "source_start_date": source_df["Date"].min().strftime("%Y-%m-%d"),
        "source_end_date": source_df["Date"].max().strftime("%Y-%m-%d"),
        "common_start_date": df["Date"].min().strftime("%Y-%m-%d"),
        "common_end_date": df["Date"].max().strftime("%Y-%m-%d"),
        "common_weeks": len(df),
        "output_file": well.output_name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gw-dir", type=Path, default=DEFAULT_GW_DIR)
    parser.add_argument("--met-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    met_dir = args.met_dir if args.met_dir is not None else default_met_dir()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_frames = {well.well_id: prepare_well(well, args.gw_dir, met_dir) for well in SELECTED_WELLS}
    cropped_frames = crop_to_common_dates(source_frames)

    summaries = []
    combined = []
    for well in SELECTED_WELLS:
        df = cropped_frames[well.well_id].copy()
        out_df = df.copy()
        out_df["Date"] = out_df["Date"].dt.strftime("%Y-%m-%d")
        out_df.to_csv(args.out_dir / well.output_name, index=False, encoding="utf-8-sig")

        long_df = out_df.copy()
        long_df.insert(0, "aquifer_type", well.aquifer_type)
        long_df.insert(1, "chinese_type", well.chinese_type)
        long_df.insert(2, "well_id", well.well_id)
        combined.append(long_df)
        summaries.append(build_summary(well, df, source_frames[well.well_id]))

    pd.DataFrame(summaries).to_csv(args.out_dir / "nine_wells_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(combined, ignore_index=True).to_csv(
        args.out_dir / "nine_wells_combined_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = pd.DataFrame(summaries)
    print(summary_df[["chinese_type", "aquifer_type", "well_id", "common_start_date", "common_end_date", "common_weeks"]])


if __name__ == "__main__":
    main()
