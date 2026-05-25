# -*- coding: utf-8 -*-
"""Run a GPU-only lookback sweep for the fixed 15-well exploratory set.

This script only uses the test split for selection summaries. future_holdout is
still produced by learn.py, but it is ignored by this sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = Path(r"C:\Users\xf-99\.conda\envs\Python39\python.exe")
REMOVED_MODELS = {"Persistence", "DynamicGatedOnly", "AdaptiveWeightedStacking"}


def parse_lookbacks(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            value = int(part)
            if value <= 0:
                raise ValueError("lookback values must be positive")
            values.append(value)
    if not values:
        raise ValueError("at least one lookback value is required")
    return values


def assert_gpu(python_exe: Path) -> None:
    code = "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
    subprocess.run([str(python_exe), "-c", code], cwd=SCRIPT_DIR, check=True)


def run_one(args: argparse.Namespace, lookback: int) -> Path:
    run_dir = args.out_dir / f"lookback_{lookback}"
    if run_dir.exists() and args.clean:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["LEARN_SKIP_VENDOR"] = "1"
    cmd = [
        str(args.python),
        str(SCRIPT_DIR / "run_15wells_test_focus.py"),
        "--lookback",
        str(lookback),
        "--seed",
        str(args.seed),
        "--out_dir",
        str(run_dir),
        "--epochs",
        str(args.epochs),
        "--gate_epochs",
        str(args.gate_epochs),
        "--mc_dropout_samples",
        str(args.mc_dropout_samples),
    ]
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)
    return run_dir


def summarize_run(run_dir: Path, lookback: int) -> pd.DataFrame:
    metrics_path = run_dir / "metrics_summary.csv"
    df = pd.read_csv(metrics_path, encoding="utf-8-sig")
    present_removed = sorted(set(df["model"].dropna()) & REMOVED_MODELS)
    if present_removed:
        raise RuntimeError(f"Removed models appeared in {metrics_path}: {present_removed}")
    df = df[df["split"].eq("test")].copy()
    df.insert(0, "lookback", lookback)
    return df


def write_summary(rows: list[pd.DataFrame], out_dir: Path) -> None:
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(out_dir / "lookback_test_metrics.csv", index=False, encoding="utf-8-sig")

    metric_cols = [c for c in ["RMSE", "MAE", "R2", "NSE"] if c in combined.columns]
    summary = (
        combined.groupby(["lookback", "model"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values(["RMSE", "lookback"])
    )
    summary.to_csv(out_dir / "lookback_test_summary.csv", index=False, encoding="utf-8-sig")

    dgs = summary[summary["model"].eq("DynamicGatedStacking")].sort_values(["RMSE", "lookback"])
    best_dgs = dgs.iloc[0].to_dict()
    overall_best = summary.iloc[0].to_dict()
    payload = {
        "selection_split": "test",
        "future_holdout_used": False,
        "seed": int(combined.attrs.get("seed", -1)),
        "best_dynamic_gated_stacking_lookback": int(best_dgs["lookback"]),
        "best_dynamic_gated_stacking_metrics": {
            k: float(v) for k, v in best_dgs.items() if k not in {"lookback", "model"} and pd.notna(v)
        },
        "overall_best_model": str(overall_best["model"]),
        "overall_best_lookback": int(overall_best["lookback"]),
        "overall_best_metrics": {
            k: float(v) for k, v in overall_best.items() if k not in {"lookback", "model"} and pd.notna(v)
        },
        "note": "Exploratory test-focused lookback sweep; not an independent unbiased test conclusion.",
    }
    (out_dir / "best_lookback_test.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    type_summary = (
        combined.groupby(["lookback", "aquifer_type", "model"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values(["lookback", "aquifer_type", "RMSE"])
    )
    type_summary.to_csv(out_dir / "lookback_test_by_type_summary.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=PYTHON_EXE)
    parser.add_argument("--lookbacks", default="12,18,24,30,36")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--gate_epochs", type=int, default=80)
    parser.add_argument("--mc_dropout_samples", type=int, default=30)
    parser.add_argument("--out_dir", type=Path, default=SCRIPT_DIR / "outputs_15wells_lookback_sweep")
    parser.add_argument("--clean", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assert_gpu(args.python)

    rows = []
    for lookback in parse_lookbacks(args.lookbacks):
        run_dir = run_one(args, lookback)
        rows.append(summarize_run(run_dir, lookback))
    combined = pd.concat(rows, ignore_index=True)
    combined.attrs["seed"] = args.seed
    write_summary([combined], args.out_dir)


if __name__ == "__main__":
    main()
