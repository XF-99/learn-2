# combo4_cand004 sharing package

This is a compact, reader-facing package for the `combo4_cand004` experiment. It keeps the code, selected data, and main evidence needed for review, while excluding logs, caches, backup files, hyperparameter-search outputs, and temporary parallel-run folders.

For the WRR-oriented manuscript story, evidence hierarchy, and wording boundaries, see:

- `docs/WRR_story_and_evidence_zh.md`

## Evidence Layout

- `code/`
  - Main training, evaluation, multi-seed, WCI/CWC, and summarization scripts.

- `data/candidate_group_cand004/`
  - The 15 selected weekly groundwater-level time-series CSV files.
  - `selected_wells_summary.csv` records the selected wells and aquifer-type labels.

- `results/01_repeatability_multiseed/`
  - Primary repeatability evidence.
  - Seeds 40-50, final point-prediction results.
  - Main files: `cand004_multiseed_model_summary.csv`, `cand004_multiseed_rank_summary.csv`.

- `results/02_point_prediction_seed45/`
  - Representative seed 45 point-prediction output.
  - Useful as a readable single-run reference, not the main repeatability evidence.

- `results/03_interval_prediction_wci_cwc_seed45/`
  - Representative seed 45 interval-prediction output.
  - Includes WCI interval calibration fields and CWC interval metrics.
  - Main file: `metrics_summary.csv`.

- `results/04_peak_prediction_seed45/`
  - Representative seed 45 peak-prediction diagnostics.
  - Includes `peak_metrics_summary.csv`.
  - `representative_examples/` keeps one representative well from each aquifer type for quick review.
  - `all_wells/` keeps peak metrics and plots for all 15 wells, so figures can be selected later.

- `results/05_shap_explainability_seed45/`
  - Representative seed 45 SHAP/explainability diagnostics.
  - `representative_examples/` keeps one representative well from each aquifer type for quick review.
  - `all_wells/` keeps SHAP/explainability figures for all 15 wells, so figures can be selected later.

- `results/06_repeatability_diagnostics_optional/`
  - Optional repeatability diagnostics from the richer seed 45 run.
  - Large raw simulation/sample tables were intentionally excluded.

- `results/results_overview.csv`
  - Compact overview of the main multi-seed point-prediction evidence and seed 45 diagnostic results.

## Main Interpretation

The primary repeatability claim is supported by `results/01_repeatability_multiseed/`, not by the interval, peak, or SHAP folders. DynamicGatedStacking is the main ensemble model because it achieved the strongest and most repeatable test-period point-prediction performance across seeds.

The interval, peak, and SHAP folders are representative diagnostics from seed 45. They are useful for uncertainty assessment, hydrological event behavior, and model interpretation, but they should not be described as full multi-seed repeatability evidence unless those analyses are rerun across all seeds.

## Reproducing Main Runs

The scripts were run on Linux with CUDA. Paths inside the shell scripts may need adjustment if this folder is moved to a different machine.

Multi-seed final point-prediction run:

```bash
bash code/run_cand004_multiseed_final.sh
```

Representative seed 45 interval/WCI run:

```bash
bash code/launch_interval_wci_seed45_parallel10.sh
```

## Notes

- This is a clean sharing package, not the full working directory.
- Excluded intentionally: `logs/`, `__pycache__/`, `.bak` files, hyperparameter-search outputs, full per-well plot folders, and `_well_parallel_tmp/`.
- The complete original working directory remains under `C:\Users\xf-99\Desktop\learn-2\yun\combo4_cand004`.
