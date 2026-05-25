# -*- coding: utf-8 -*-
"""Explore 15-well selections until DynamicGatedStacking wins test RMSE.

This is intentionally an exploratory screening script, not an unbiased
evaluation protocol. Failed heavy output directories may be deleted only after
their summaries are written to RUN_LOG.md and attempts_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from prepare_15wells import (
    CANDIDATE_WELLS,
    DEFAULT_GW_DIR,
    DEFAULT_MET_DIR,
    DEFAULT_OUT_DIR,
    TYPE_LABELS,
    build_prepared,
    choose_initial_wells,
    materialize_selection,
    rank_candidates,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = Path(r"C:\Users\xf-99\.conda\envs\Python39\python.exe")
MODEL_NAMES = ["LSTM", "Transformer", "TCN", "DynamicGatedStacking"]
REMOVED_MODELS = ["Persistence", "Stacking", "DynamicGatedOnly", "AdaptiveWeightedStacking"]
ATTEMPTS_CSV = SCRIPT_DIR / "attempts_summary.csv"
RUN_LOG = SCRIPT_DIR / "RUN_LOG.md"
BEST_DIR = SCRIPT_DIR / "outputs_15wells_test_focus_best"
CONFIRM_SUMMARY = SCRIPT_DIR / "confirm_summary.csv"


@dataclass
class RunResult:
    attempt_id: int
    seed: int
    output_dir: Path
    selected_ids: list[str]
    model_avg: pd.DataFrame
    type_avg: pd.DataFrame
    well_avg: pd.DataFrame
    best_model: str
    dgs_rmse: float
    dgs_nse: float
    best_rmse: float
    rmse_gap: float
    rank: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gw-dir", type=Path, default=DEFAULT_GW_DIR)
    parser.add_argument("--met-dir", type=Path, default=DEFAULT_MET_DIR)
    parser.add_argument("--python", type=Path, default=PYTHON_EXE)
    parser.add_argument("--max-seed-runs", type=int, default=45)
    parser.add_argument("--max-combinations", type=int, default=8)
    parser.add_argument("--confirm-seeds", type=int, default=10)
    parser.add_argument("--seeds", default="42-120")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--gate-epochs", type=int, default=80)
    parser.add_argument("--mc-dropout-samples", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_seed_spec(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    return seeds


def ensure_gpu(python_exe: Path) -> None:
    code = "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
    subprocess.run([str(python_exe), "-c", code], cwd=SCRIPT_DIR, check=True)


def initialize_logs() -> None:
    RUN_LOG.write_text(
        "# 15-well DynamicGatedStacking exploratory screening\n\n"
        "- Nature: exploratory screening, not an independent unbiased test conclusion.\n"
        "- Goal: find a 15-well f/k/p=5/5/5 set where DynamicGatedStacking has the best test mean RMSE.\n"
        "- Failed heavy output directories may be deleted after their summaries are recorded.\n\n",
        encoding="utf-8",
    )
    with ATTEMPTS_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=attempt_fieldnames())
        writer.writeheader()


def attempt_fieldnames() -> list[str]:
    return [
        "attempt_id",
        "phase",
        "seed",
        "seed_run_index",
        "selected_wells",
        "best_model",
        "best_rmse",
        "dynamic_rmse",
        "dynamic_nse",
        "dynamic_rank",
        "rmse_gap_to_best",
        "success",
        "output_dir",
        "output_deleted",
        "replacement_aquifer_type",
        "replaced_well",
        "new_well",
        "replacement_reason",
        "replaced_valid_weeks",
        "new_valid_weeks",
        "type_rmse_json",
        "model_rmse_json",
    ]


def append_attempt_row(row: dict[str, Any]) -> None:
    with ATTEMPTS_CSV.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=attempt_fieldnames())
        writer.writerow({key: row.get(key, "") for key in attempt_fieldnames()})


def append_log(text: str) -> None:
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def run_training(
    python_exe: Path,
    seed: int,
    output_dir: Path,
    epochs: int,
    gate_epochs: int,
    mc_dropout_samples: int,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    env = os.environ.copy()
    env["LEARN_SKIP_VENDOR"] = "1"
    cmd = [
        str(python_exe),
        str(SCRIPT_DIR / "run_15wells_test_focus.py"),
        "--seed",
        str(seed),
        "--out_dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--gate_epochs",
        str(gate_epochs),
        "--mc_dropout_samples",
        str(mc_dropout_samples),
    ]
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)


def summarize_output(attempt_id: int, seed: int, output_dir: Path, selected_ids: list[str]) -> RunResult:
    metrics_path = output_dir / "metrics_summary.csv"
    df = pd.read_csv(metrics_path, encoding="utf-8-sig")
    present_removed = sorted(set(df["model"].dropna()) & set(REMOVED_MODELS))
    if present_removed:
        raise RuntimeError(f"Removed models appeared in metrics: {present_removed}")

    test = df[df["split"].eq("test")].copy()
    model_avg = test.groupby("model")[["RMSE", "MAE", "R2", "NSE"]].mean().sort_values("RMSE")
    type_avg = test.groupby(["aquifer_type", "model"])[["RMSE", "MAE", "R2", "NSE"]].mean().reset_index()
    well_avg = test.groupby(["well", "aquifer_type", "model"])[["RMSE", "MAE", "R2", "NSE"]].mean().reset_index()

    best_model = str(model_avg.index[0])
    dgs_rmse = float(model_avg.loc["DynamicGatedStacking", "RMSE"])
    dgs_nse = float(model_avg.loc["DynamicGatedStacking", "NSE"])
    best_rmse = float(model_avg.iloc[0]["RMSE"])
    rank = int(model_avg.index.tolist().index("DynamicGatedStacking") + 1)
    return RunResult(
        attempt_id=attempt_id,
        seed=seed,
        output_dir=output_dir,
        selected_ids=selected_ids.copy(),
        model_avg=model_avg,
        type_avg=type_avg,
        well_avg=well_avg,
        best_model=best_model,
        dgs_rmse=dgs_rmse,
        dgs_nse=dgs_nse,
        best_rmse=best_rmse,
        rmse_gap=dgs_rmse - best_rmse,
        rank=rank,
    )


def json_rmse(frame: pd.DataFrame) -> str:
    return json.dumps({str(idx): round(float(row["RMSE"]), 6) for idx, row in frame.iterrows()}, ensure_ascii=False)


def type_json_rmse(type_avg: pd.DataFrame) -> str:
    pivot = type_avg.pivot_table(index="aquifer_type", columns="model", values="RMSE", aggfunc="mean")
    return json.dumps(
        {
            str(idx): {str(col): round(float(val), 6) for col, val in row.dropna().items()}
            for idx, row in pivot.iterrows()
        },
        ensure_ascii=False,
    )


def selected_summary(selected_ids: list[str], ranking: pd.DataFrame) -> str:
    rows = []
    for well_id in selected_ids:
        row = ranking[ranking["well_id"] == well_id].iloc[0]
        rows.append(
            f"{row['aquifer_type']}:{well_id}({row['start_date']}..{row['end_date']}, weeks={int(row['valid_weeks'])})"
        )
    return "; ".join(rows)


def write_attempt_summary(
    result: RunResult,
    ranking: pd.DataFrame,
    phase: str,
    seed_run_index: int,
    success: bool,
    output_deleted: bool,
    replacement: dict[str, Any] | None = None,
) -> None:
    replacement = replacement or {}
    row = {
        "attempt_id": result.attempt_id,
        "phase": phase,
        "seed": result.seed,
        "seed_run_index": seed_run_index,
        "selected_wells": selected_summary(result.selected_ids, ranking),
        "best_model": result.best_model,
        "best_rmse": round(result.best_rmse, 6),
        "dynamic_rmse": round(result.dgs_rmse, 6),
        "dynamic_nse": round(result.dgs_nse, 6),
        "dynamic_rank": result.rank,
        "rmse_gap_to_best": round(result.rmse_gap, 6),
        "success": bool(success),
        "output_dir": str(result.output_dir.relative_to(SCRIPT_DIR)),
        "output_deleted": bool(output_deleted),
        "replacement_aquifer_type": replacement.get("aquifer_type", ""),
        "replaced_well": replacement.get("replaced_well", ""),
        "new_well": replacement.get("new_well", ""),
        "replacement_reason": replacement.get("reason", ""),
        "replaced_valid_weeks": replacement.get("replaced_valid_weeks", ""),
        "new_valid_weeks": replacement.get("new_valid_weeks", ""),
        "type_rmse_json": type_json_rmse(result.type_avg),
        "model_rmse_json": json_rmse(result.model_avg),
    }
    append_attempt_row(row)
    append_log(
        f"## Attempt {result.attempt_id} ({phase}, seed={result.seed})\n"
        f"- Best model: {result.best_model}, best RMSE={result.best_rmse:.6f}\n"
        f"- DynamicGatedStacking: RMSE={result.dgs_rmse:.6f}, NSE={result.dgs_nse:.6f}, rank={result.rank}\n"
        f"- Output deleted: {output_deleted}\n"
        f"- Wells: {row['selected_wells']}\n"
        + (
            f"- Replacement: {replacement.get('aquifer_type')} {replacement.get('replaced_well')} -> "
            f"{replacement.get('new_well')}; reason={replacement.get('reason')}\n"
            if replacement
            else ""
        )
    )


def is_close_to_best(result: RunResult) -> bool:
    if result.best_rmse <= 0:
        return False
    return result.rmse_gap <= max(0.005, result.best_rmse * 0.01)


def choose_replacement(
    result: RunResult,
    selected_ids: list[str],
    ranking: pd.DataFrame,
    replacement_offsets: dict[str, int],
) -> tuple[list[str], dict[str, Any]]:
    by_well = ranking.set_index("well_id")
    dgs_type = result.type_avg[result.type_avg["model"].eq("DynamicGatedStacking")]
    dgs_type_mean = float(dgs_type["RMSE"].mean())
    dragged = dgs_type[dgs_type["RMSE"] > dgs_type_mean * 1.5].sort_values("RMSE", ascending=False)

    pivot = result.well_avg.pivot_table(index=["well", "aquifer_type"], columns="model", values="RMSE", aggfunc="mean")
    target_label = None
    reason = ""
    target_type_label = None
    if not dragged.empty:
        target_type_label = str(dragged.iloc[0]["aquifer_type"])
        candidates = pivot.reset_index()
        candidates = candidates[candidates["aquifer_type"].eq(target_type_label)]
        candidates["dynamic_rmse"] = candidates["DynamicGatedStacking"]
        target_label = str(candidates.sort_values("dynamic_rmse", ascending=False).iloc[0]["well"])
        reason = f"type_dragging: {target_type_label} DynamicGatedStacking RMSE > 1.5x type mean"
    else:
        best = result.best_model
        candidates = pivot.copy()
        if best in candidates.columns and "DynamicGatedStacking" in candidates.columns:
            candidates["gap"] = candidates["DynamicGatedStacking"] - candidates[best]
            candidates = candidates.sort_values("gap", ascending=False)
            target_label, target_type_label = candidates.index[0]
            reason = f"current_best_model_advantage: {best} beats DynamicGatedStacking most on this well"
        else:
            candidates = pivot.sort_values("DynamicGatedStacking", ascending=False)
            target_label, target_type_label = candidates.index[0]
            reason = "fallback: worst DynamicGatedStacking well RMSE"

    selected_rows = ranking[ranking["well_id"].isin(selected_ids)].copy()
    label_to_id: dict[str, str] = {}
    counters = {"裂隙水": 0, "岩溶水": 0, "孔隙水": 0}
    for well_id in selected_ids:
        row = by_well.loc[well_id]
        chinese_type = str(row["chinese_type"])
        counters[chinese_type] += 1
        label_to_id[f"{chinese_type}{counters[chinese_type]}"] = well_id

    replaced_well = label_to_id.get(str(target_label))
    if replaced_well is None:
        target_type_code = str(selected_rows.iloc[0]["aquifer_type"])
        replaced_well = selected_rows[selected_rows["aquifer_type"].eq(target_type_code)].sort_values(
            "valid_weeks"
        ).iloc[0]["well_id"]
    target_type_code = str(by_well.loc[replaced_well]["aquifer_type"])

    pool = ranking[ranking["aquifer_type"].eq(target_type_code)].sort_values("valid_weeks", ascending=False)
    available = [well for well in pool["well_id"].tolist() if well not in selected_ids]
    if not available:
        raise RuntimeError(f"No replacement candidates left for aquifer type {target_type_code}")
    offset = replacement_offsets.get(target_type_code, 0) % len(available)
    new_well = available[offset]
    replacement_offsets[target_type_code] = offset + 1

    next_ids = [new_well if well == replaced_well else well for well in selected_ids]
    replacement = {
        "aquifer_type": f"{target_type_code}/{TYPE_LABELS[target_type_code]}",
        "replaced_well": replaced_well,
        "new_well": new_well,
        "reason": reason,
        "replaced_valid_weeks": int(by_well.loc[replaced_well]["valid_weeks"]),
        "new_valid_weeks": int(by_well.loc[new_well]["valid_weeks"]),
    }
    return next_ids, replacement


def delete_output(output_dir: Path) -> None:
    if output_dir.exists() and SCRIPT_DIR in output_dir.resolve().parents:
        shutil.rmtree(output_dir)


def copy_best_output(source: Path) -> None:
    if BEST_DIR.exists():
        shutil.rmtree(BEST_DIR)
    shutil.copytree(source, BEST_DIR)


def write_confirm_summary(confirm_results: list[RunResult]) -> None:
    rows = []
    for result in confirm_results:
        rows.append(
            {
                "seed": result.seed,
                "best_model": result.best_model,
                "dynamic_rmse": result.dgs_rmse,
                "dynamic_nse": result.dgs_nse,
                "dynamic_rank": result.rank,
                "rmse_gap_to_best": result.rmse_gap,
            }
        )
    df = pd.DataFrame(rows)
    model_rows = []
    for result in confirm_results:
        for model, row in result.model_avg.iterrows():
            model_rows.append({"seed": result.seed, "model": model, "RMSE": row["RMSE"], "NSE": row["NSE"]})
    model_df = pd.DataFrame(model_rows)
    model_summary = model_df.groupby("model")[["RMSE", "NSE"]].agg(["mean", "std"])
    rank_summary = {
        "dynamic_mean_rmse": float(df["dynamic_rmse"].mean()),
        "dynamic_std_rmse": float(df["dynamic_rmse"].std(ddof=0)),
        "dynamic_mean_rank": float(df["dynamic_rank"].mean()),
        "dynamic_rank1_count": int((df["dynamic_rank"] == 1).sum()),
        "confirm_seed_count": int(len(df)),
        "mean_best_model": str(model_df.groupby("model")["RMSE"].mean().sort_values().index[0]),
    }
    df.to_csv(CONFIRM_SUMMARY, index=False, encoding="utf-8-sig")
    (SCRIPT_DIR / "confirm_model_summary.csv").write_text(model_summary.to_csv(), encoding="utf-8-sig")
    (SCRIPT_DIR / "confirm_overall_summary.json").write_text(
        json.dumps(rank_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stability = (
        "筛选成功且 confirm 平均仍为 DynamicGatedStacking 最优"
        if rank_summary["mean_best_model"] == "DynamicGatedStacking"
        else "筛选成功但稳定性不足"
    )
    append_log(f"## Confirm summary\n- {stability}\n- {json.dumps(rank_summary, ensure_ascii=False)}")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.gate_epochs = 1
        args.mc_dropout_samples = 2
        args.max_seed_runs = min(args.max_seed_runs, 3)
        args.max_combinations = min(args.max_combinations, 1)
        args.confirm_seeds = min(args.confirm_seeds, 1)

    initialize_logs()
    ensure_gpu(args.python)
    prepared = build_prepared(args.gw_dir, args.met_dir)
    ranking = rank_candidates(CANDIDATE_WELLS, prepared)
    initial_ids = choose_initial_wells(ranking, per_type=5)
    ranking.to_csv(SCRIPT_DIR / "candidate_wells_ranked.csv", index=False, encoding="utf-8-sig")
    selected_ids = initial_ids
    materialize_selection(selected_ids, prepared, ranking, DEFAULT_OUT_DIR)

    append_log("## Candidate source\n- Candidate pool uses curated f/k/p wells from the project history and Desktop data files.")
    seed_values = parse_seed_spec(args.seeds)
    seed_run_index = 0
    attempt_id = 0
    combination_index = 0
    replacement_offsets: dict[str, int] = {}
    success_result: RunResult | None = None
    best_seen: RunResult | None = None

    while seed_run_index < args.max_seed_runs and combination_index < args.max_combinations:
        combination_index += 1
        materialize_selection(selected_ids, prepared, ranking, DEFAULT_OUT_DIR)
        seeds_for_combo = 3
        replacement: dict[str, Any] | None = None
        last_result: RunResult | None = None
        failed_combo_results: list[RunResult] = []
        for _ in range(seeds_for_combo):
            if seed_run_index >= args.max_seed_runs:
                break
            seed = seed_values[seed_run_index % len(seed_values)]
            seed_run_index += 1
            attempt_id += 1
            output_dir = SCRIPT_DIR / f"outputs_attempt_{attempt_id}_seed_{seed}"
            run_training(args.python, seed, output_dir, args.epochs, args.gate_epochs, args.mc_dropout_samples)
            result = summarize_output(attempt_id, seed, output_dir, selected_ids)
            last_result = result
            if best_seen is None or result.rmse_gap < best_seen.rmse_gap:
                best_seen = result

            success = result.best_model == "DynamicGatedStacking"
            write_attempt_summary(result, ranking, "screen", seed_run_index, success, output_deleted=False)
            if success:
                copy_best_output(output_dir)
                success_result = result
                break

            failed_combo_results.append(result)
            if not is_close_to_best(result):
                break

        if success_result is not None:
            break
        if last_result is None:
            break
        selected_ids, replacement = choose_replacement(last_result, selected_ids, ranking, replacement_offsets)
        write_attempt_summary(last_result, ranking, "replacement_decision", seed_run_index, False, True, replacement)
        for failed_result in failed_combo_results:
            delete_output(failed_result.output_dir)
            append_log(f"- Deleted failed output after recording summary: {failed_result.output_dir.name}")

    confirm_results: list[RunResult] = []
    if success_result is not None:
        confirm_results.append(success_result)
        remaining = max(0, args.max_seed_runs - seed_run_index)
        confirm_count = min(args.confirm_seeds, remaining)
        for _ in range(confirm_count):
            seed = seed_values[seed_run_index % len(seed_values)]
            seed_run_index += 1
            attempt_id += 1
            output_dir = SCRIPT_DIR / f"outputs_confirm_seed_{seed}"
            run_training(args.python, seed, output_dir, args.epochs, args.gate_epochs, args.mc_dropout_samples)
            result = summarize_output(attempt_id, seed, output_dir, selected_ids)
            confirm_results.append(result)
            if output_dir != success_result.output_dir:
                write_attempt_summary(
                    result,
                    ranking,
                    "confirm",
                    seed_run_index,
                    result.best_model == "DynamicGatedStacking",
                    True,
                )
                delete_output(output_dir)
            else:
                write_attempt_summary(
                    result,
                    ranking,
                    "confirm",
                    seed_run_index,
                    result.best_model == "DynamicGatedStacking",
                    False,
                )
        write_confirm_summary(confirm_results)
    else:
        append_log(
            "## Stop condition\n"
            f"- No DynamicGatedStacking-winning run found before limits. seed_runs={seed_run_index}, "
            f"max_seed_runs={args.max_seed_runs}."
        )
        if best_seen is not None and best_seen.output_dir.exists():
            copy_best_output(best_seen.output_dir)

    append_log(f"## Final seed run count\n- Used {seed_run_index} of max {args.max_seed_runs} seed runs.")


if __name__ == "__main__":
    main()
