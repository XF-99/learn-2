# -*- coding: utf-8 -*-
"""Run fixed 15-well test-focused experiments across multiple random seeds."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEEDS = "45,46,47,48,49,50,51,52,53,54"


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def run_seed(args: argparse.Namespace, seed: int) -> Path:
    seed_dir = args.out_dir / f"seed_{seed}"
    if seed_dir.exists() and args.clean:
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["LEARN_SKIP_VENDOR"] = "1"
    cmd = [
        str(args.python),
        str(SCRIPT_DIR / "run_15wells_test_focus.py"),
        "--seed",
        str(seed),
        "--out_dir",
        str(seed_dir),
        "--epochs",
        str(args.epochs),
        "--gate_epochs",
        str(args.gate_epochs),
        "--mc_dropout_samples",
        str(args.mc_dropout_samples),
    ]
    if args.lookback is not None:
        cmd.extend(["--lookback", str(args.lookback)])
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)
    return seed_dir


def collect_metrics(out_dir: Path, seeds: list[int]) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = out_dir / f"seed_{seed}" / "metrics_summary.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def collect_predictions(out_dir: Path, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}"
        for well_dir in sorted(path for path in seed_dir.iterdir() if path.is_dir()):
            prediction_path = well_dir / "test_predictions.csv"
            if not prediction_path.exists():
                continue
            frame = pd.read_csv(prediction_path)
            frame.insert(0, "well", well_dir.name)
            frame.insert(0, "seed", seed)
            rows.append(frame)
    if not rows:
        raise ValueError(f"No test_predictions.csv files found under {out_dir}.")
    return pd.concat(rows, ignore_index=True)


def write_summaries(out_dir: Path, seeds: list[int]) -> None:
    metrics = collect_metrics(out_dir, seeds)
    metrics.to_csv(out_dir / "multiseed_metrics_summary.csv", index=False, encoding="utf-8-sig")

    test_metrics = metrics[metrics["split"].eq("test")].copy()
    metric_cols = [col for col in ["RMSE", "MAE", "R2", "NSE"] if col in test_metrics.columns]
    by_model = (
        test_metrics.groupby(["model"], as_index=False)[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    by_model.columns = ["_".join(str(part) for part in col if part) for col in by_model.columns.to_flat_index()]
    by_model.to_csv(out_dir / "multiseed_test_model_summary.csv", index=False, encoding="utf-8-sig")

    predictions = collect_predictions(out_dir, seeds)
    predictions.to_csv(out_dir / "multiseed_test_predictions_long.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--out_dir", type=Path, default=SCRIPT_DIR / "outputs_15wells_multiseed_test_focus")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--gate_epochs", type=int, default=80)
    parser.add_argument("--mc_dropout_samples", type=int, default=30)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_runs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    if not args.skip_runs:
        for seed in seeds:
            run_seed(args, seed)
    write_summaries(args.out_dir, seeds)


if __name__ == "__main__":
    main()
