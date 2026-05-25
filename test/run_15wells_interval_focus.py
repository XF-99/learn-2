# -*- coding: utf-8 -*-
"""GPU-only interval-prediction entrypoint for the fixed 15-well set."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARGS = [
    "--wells_dir",
    "selected_weekly_data_15wells_current",
    "--disable_explain",
    "--disable_peak_analysis",
    "--lookback",
    "18",
    "--holdout_steps",
    "1",
    "--train_ratio",
    "0.60",
    "--val_ratio",
    "0.10",
    "--selection_ratio",
    "0.05",
    "--calib_ratio",
    "0.20",
    "--epochs",
    "60",
    "--gate_epochs",
    "80",
    "--mc_dropout_samples",
    "30",
    "--batch_size",
    "128",
    "--interval_width_scale",
    "1.15",
    "--out_dir",
    "outputs_15wells_interval_focus",
]


def assert_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("GPU is required for this experiment, but torch.cuda.is_available() is false.")


def main() -> None:
    os.environ.setdefault("LEARN_SKIP_VENDOR", "1")
    os.chdir(SCRIPT_DIR)
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    assert_gpu()
    sys.argv = ["learn.py", *DEFAULT_ARGS, *sys.argv[1:]]
    runpy.run_module("learn", run_name="__main__")


if __name__ == "__main__":
    main()
