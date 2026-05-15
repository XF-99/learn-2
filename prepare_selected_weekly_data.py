# -*- coding: utf-8 -*-
"""Prepare weekly groundwater and meteorological datasets for selected wells."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_GW_DIR = Path(r"C:\Users\xf-99\Downloads\GWData (1)\GWData")
DEFAULT_MET_DIR = Path(r"C:\Users\xf-99\Desktop\气象数据\气象数据")
DEFAULT_OUT_DIR = WORKSPACE / "selected_weekly_data"

FIELD_ORDER = ["Date", "TASMAX", "TAS", "TASMIN", "Humidity", "Precipitation", "GWL"]


@dataclass(frozen=True)
class WellSelection:
    aquifer_type: str
    chinese_type: str
    well_id: str
    name: str
    x_coord: int
    y_coord: int
    ground_surface_m_asl: float
    depth_to_gw_m: float
    output_name: str


@dataclass(frozen=True)
class MeteoSpec:
    folder: str
    column: str
    output_column: str
    agg: str
    filename_suffix: str


SELECTED_WELLS = [
    WellSelection(
        aquifer_type="f",
        chinese_type="裂隙水",
        well_id="HE_6253",
        name="NETRA",
        x_coord=576396,
        y_coord=5661004,
        ground_surface_m_asl=312.7,
        depth_to_gw_m=8.86,
        output_name="裂隙水_HE_6253_每周数据.csv",
    ),
    WellSelection(
        aquifer_type="k",
        chinese_type="岩溶水",
        well_id="BY_15120",
        name="IHRLERSTEIN TIEF K1",
        x_coord=707357,
        y_coord=5426437,
        ground_surface_m_asl=480.03,
        depth_to_gw_m=94.03,
        output_name="岩溶水_BY_15120_每周数据.csv",
    ),
    WellSelection(
        aquifer_type="p",
        chinese_type="孔隙水",
        well_id="SN_46460564",
        name="Walda",
        x_coord=813164,
        y_coord=5695042,
        ground_surface_m_asl=109.22,
        depth_to_gw_m=1.90,
        output_name="孔隙水_SN_46460564_每周数据.csv",
    ),
]


METEO_SPECS = [
    MeteoSpec("最高气温", "TASMAX_C", "TASMAX", "max", "最高温度数据.csv"),
    MeteoSpec("平均气温", "TAS_C", "TAS", "mean", "平均温度数据.csv"),
    MeteoSpec("最低气温", "TASMIN_C", "TASMIN", "min", "最低温度数据.csv"),
    MeteoSpec("湿度", "Humidity_Percent", "Humidity", "mean", "相对湿度数据.csv"),
    MeteoSpec("降水", "Precipitation_mm", "Precipitation", "sum", "降水数据.csv"),
]


def read_groundwater(gw_dir: Path, well_id: str) -> pd.DataFrame:
    path = gw_dir / f"{well_id}_GW-Data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing groundwater file: {path}")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["GWL"] = pd.to_numeric(df["GWL"], errors="coerce")
    df = df.dropna(subset=["Date", "GWL"]).sort_values("Date")
    return df[["Date", "GWL"]].drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)


def read_meteo_daily(met_dir: Path, well_id: str, spec: MeteoSpec) -> pd.DataFrame:
    path = met_dir / spec.folder / f"{well_id}_{spec.filename_suffix}"
    if not path.exists():
        raise FileNotFoundError(f"Missing meteorological file: {path}")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[spec.column] = pd.to_numeric(df[spec.column], errors="coerce")
    df = df.dropna(subset=["Date", spec.column]).sort_values("Date")
    return df[["Date", spec.column]].drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)


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
            value_column=spec.column,
            agg=spec.agg,
        )

    result = result[FIELD_ORDER].dropna(subset=FIELD_ORDER).sort_values("Date").reset_index(drop=True)
    result["Date"] = result["Date"].dt.strftime("%Y-%m-%d")
    return result


def build_summary(well: WellSelection, df: pd.DataFrame) -> dict[str, object]:
    return {
        "well_id": well.well_id,
        "name": well.name,
        "aquifer_type": well.aquifer_type,
        "chinese_type": well.chinese_type,
        "x_coord_utm32n": well.x_coord,
        "y_coord_utm32n": well.y_coord,
        "ground_surface_m_asl": well.ground_surface_m_asl,
        "depth_to_gw_m": well.depth_to_gw_m,
        "start_date": df["Date"].min(),
        "end_date": df["Date"].max(),
        "valid_weeks": len(df),
        "output_file": well.output_name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gw-dir", type=Path, default=DEFAULT_GW_DIR)
    parser.add_argument("--met-dir", type=Path, default=DEFAULT_MET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for well in SELECTED_WELLS:
        df = prepare_well(well, args.gw_dir, args.met_dir)
        if df.empty:
            raise ValueError(f"No valid weekly rows produced for {well.well_id}")
        df.to_csv(args.out_dir / well.output_name, index=False, encoding="utf-8-sig")
        summaries.append(build_summary(well, df))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.out_dir / "selected_wells_summary.csv", index=False, encoding="utf-8-sig")

    for item in summaries:
        print(
            f"{item['chinese_type']} {item['well_id']} {item['name']}: "
            f"{item['start_date']} to {item['end_date']}, {item['valid_weeks']} weeks"
        )


if __name__ == "__main__":
    main()
