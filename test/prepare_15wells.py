# -*- coding: utf-8 -*-
"""Prepare candidate groundwater wells for the 15-well exploratory run.

The script ranks known f/k/p candidates by the number of valid weekly rows after
joining groundwater observations with meteorological variables. It can then
materialize any selected set into the format consumed by learn.py.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GW_DIR = Path(r"C:\Users\xf-99\Desktop\GWData (1)\GWData")
DEFAULT_MET_DIR = Path(r"C:\Users\xf-99\Desktop\气象数据\气象数据")
DEFAULT_OUT_DIR = SCRIPT_DIR / "selected_weekly_data_15wells_current"
FIELD_ORDER = ["Date", "TASMAX", "TAS", "TASMIN", "Humidity", "Precipitation", "GWL"]
TYPE_LABELS = {"f": "裂隙水", "k": "岩溶水", "p": "孔隙水"}


@dataclass(frozen=True)
class WellCandidate:
    aquifer_type: str
    well_id: str
    name: str
    source: str = "curated_pool"

    @property
    def chinese_type(self) -> str:
        return TYPE_LABELS[self.aquifer_type.lower()]

    @property
    def output_name(self) -> str:
        return f"{self.aquifer_type.lower()}_{self.well_id}_weekly.csv"


@dataclass(frozen=True)
class MeteoSpec:
    source_column: str
    output_column: str
    agg: str


BASELINE_WELLS = [
    WellCandidate("f", "HE_6253", "NETRA", "project_history"),
    WellCandidate("f", "HE_12117", "HE_12117", "project_history"),
    WellCandidate("f", "HE_7824", "HE_7824", "project_history"),
    WellCandidate("k", "BY_15120", "IHRLERSTEIN TIEF K1", "project_history"),
    WellCandidate("k", "BY_11119", "BY_11119", "project_history"),
    WellCandidate("k", "BY_7126", "BY_7126", "project_history"),
    WellCandidate("p", "SN_46460564", "Walda", "project_history"),
    WellCandidate("p", "SN_49430964", "SN_49430964", "project_history"),
    WellCandidate("p", "SN_49484004", "SN_49484004", "project_history"),
]


CANDIDATE_WELLS = [
    *BASELINE_WELLS,
    WellCandidate("f", "BY_83614", "NBS-H_W KB 11_1"),
    WellCandidate("f", "HE_10319", "LETTGENBRUNN"),
    WellCandidate("f", "NI_100000842", "Ehmen II"),
    WellCandidate("f", "NI_100000926", "Sehlde"),
    WellCandidate("f", "NW_100140762", "WG 70 TAPPENAU"),
    WellCandidate("f", "SN_52410759", "Muelsen-St-Niclas"),
    WellCandidate("f", "ST_44339213", "Lengefeld"),
    WellCandidate("k", "BW_100-813-7", "GIENGEN TAUBENTAL"),
    WellCandidate("k", "BW_103-763-0", "Sontheimer Wirtshaeusle, STEINHEIM"),
    WellCandidate("k", "BY_24153", "SPEINSHART Q3"),
    WellCandidate("k", "NW_129660176", "Silbecke"),
    WellCandidate("k", "NW_129660206", "Schoenholthausen I"),
    WellCandidate("k", "NW_91163705", "Poeppelsche Eikeloh"),
    WellCandidate("k", "NW_91174909", "Brilon LederkeOL748"),
    WellCandidate("p", "BB_32455305", "Hohenbruch, Weg n.Teerofen"),
    WellCandidate("p", "NW_100140142", "WG 22 LEVKENSTAD"),
    WellCandidate("p", "NW_60090169", "HS 67"),
    WellCandidate("p", "NW_80000186", "OEDT Nr020"),
    WellCandidate("p", "RP_2378140100", "1057 Boebingen"),
]


METEO_SPECS = [
    MeteoSpec("TASMAX_C", "TASMAX", "max"),
    MeteoSpec("TAS_C", "TAS", "mean"),
    MeteoSpec("TASMIN_C", "TASMIN", "min"),
    MeteoSpec("Humidity_Percent", "Humidity", "mean"),
    MeteoSpec("Precipitation_mm", "Precipitation", "sum"),
]


def unique_candidates() -> list[WellCandidate]:
    candidates: dict[str, WellCandidate] = {}
    for candidate in CANDIDATE_WELLS:
        candidates.setdefault(candidate.well_id, candidate)
    return list(candidates.values())


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
    matches: list[Path] = []
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
        window = daily_series.loc[obs_date - pd.Timedelta(days=6) : obs_date]
        values.append(float(reducer(window)) if len(window) == 7 else pd.NA)
    return pd.Series(values, index=observation_dates.index)


def prepare_well(candidate: WellCandidate, gw_dir: Path, met_dir: Path) -> pd.DataFrame:
    result = read_groundwater(gw_dir, candidate.well_id)
    for spec in METEO_SPECS:
        daily = read_meteo_daily(met_dir, candidate.well_id, spec)
        result[spec.output_column] = aggregate_for_observation_dates(
            observation_dates=result["Date"],
            daily=daily,
            value_column=spec.source_column,
            agg=spec.agg,
        )
    return result[FIELD_ORDER].dropna(subset=FIELD_ORDER).sort_values("Date").reset_index(drop=True)


def build_prepared(gw_dir: Path, met_dir: Path) -> dict[str, pd.DataFrame]:
    prepared: dict[str, pd.DataFrame] = {}
    for candidate in unique_candidates():
        try:
            prepared[candidate.well_id] = prepare_well(candidate, gw_dir, met_dir)
        except Exception as exc:
            print(f"SKIP {candidate.well_id}: {exc}")
    return prepared


def rank_candidates(candidates: Iterable[WellCandidate], prepared: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.well_id in seen or candidate.well_id not in prepared:
            continue
        seen.add(candidate.well_id)
        df = prepared[candidate.well_id]
        rows.append(
            {
                "aquifer_type": candidate.aquifer_type.lower(),
                "chinese_type": candidate.chinese_type,
                "well_id": candidate.well_id,
                "name": candidate.name,
                "source": candidate.source,
                "start_date": df["Date"].min().strftime("%Y-%m-%d"),
                "end_date": df["Date"].max().strftime("%Y-%m-%d"),
                "valid_weeks": int(len(df)),
            }
        )
    return pd.DataFrame(rows).sort_values(["aquifer_type", "valid_weeks"], ascending=[True, False])


def choose_initial_wells(ranking: pd.DataFrame, per_type: int) -> list[str]:
    chosen: list[str] = []
    for aquifer_type in ["f", "k", "p"]:
        group = ranking[ranking["aquifer_type"].str.lower() == aquifer_type].sort_values(
            "valid_weeks", ascending=False
        )
        if len(group) < per_type:
            raise ValueError(f"Only {len(group)} candidates for aquifer type {aquifer_type}")
        chosen.extend(group.head(per_type)["well_id"].tolist())
    return chosen


def materialize_selection(
    selected_ids: Iterable[str],
    prepared: dict[str, pd.DataFrame],
    ranking: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_id = {candidate.well_id: candidate for candidate in unique_candidates()}
    selected_rows = []
    combined = []
    counters = {"f": 0, "k": 0, "p": 0}
    for well_id in selected_ids:
        candidate = by_id[well_id]
        aquifer_type = candidate.aquifer_type.lower()
        counters[aquifer_type] += 1
        label = f"{candidate.chinese_type}{counters[aquifer_type]}"

        df = prepared[well_id].copy()
        out_df = df.copy()
        out_df["Date"] = out_df["Date"].dt.strftime("%Y-%m-%d")
        out_file = candidate.output_name
        out_df.to_csv(out_dir / out_file, index=False, encoding="utf-8-sig")

        row = ranking[ranking["well_id"] == well_id].iloc[0].to_dict()
        row.update({"label": label, "output_file": out_file})
        selected_rows.append(row)

        long_df = out_df.copy()
        long_df.insert(0, "aquifer_type", aquifer_type)
        long_df.insert(1, "chinese_type", candidate.chinese_type)
        long_df.insert(2, "well_id", well_id)
        long_df.insert(3, "label", label)
        combined.append(long_df)

    selected = pd.DataFrame(selected_rows)
    selected.to_csv(out_dir / "selected_wells_summary.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(out_dir / "candidate_wells_ranked.csv", index=False, encoding="utf-8-sig")
    pd.concat(combined, ignore_index=True).to_csv(
        out_dir / "selected_wells_combined_long.csv", index=False, encoding="utf-8-sig"
    )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gw-dir", type=Path, default=DEFAULT_GW_DIR)
    parser.add_argument("--met-dir", type=Path, default=DEFAULT_MET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--per-type", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = build_prepared(args.gw_dir, args.met_dir)
    ranking = rank_candidates(CANDIDATE_WELLS, prepared)
    selected_ids = choose_initial_wells(ranking, args.per_type)
    selected = materialize_selection(selected_ids, prepared, ranking, args.out_dir)
    print(selected[["chinese_type", "aquifer_type", "well_id", "start_date", "end_date", "valid_weeks"]].to_string(index=False))


if __name__ == "__main__":
    main()
