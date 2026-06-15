# Results Folder

This folder is organized by evidence level.

- `01_repeatability_multiseed/`: primary repeatability evidence from seeds 40-50.
- `02_point_prediction_seed45/`: representative seed 45 point-prediction output.
- `03_interval_prediction_wci_cwc_seed45/`: representative seed 45 interval prediction with WCI/CWC fields.
- `04_peak_prediction_seed45/`: representative seed 45 peak-prediction diagnostics. Use `representative_examples/` for quick review and `all_wells/` for all 15 wells.
- `05_shap_explainability_seed45/`: representative seed 45 SHAP/explainability diagnostics. Use `representative_examples/` for quick review and `all_wells/` for all 15 wells.
- `06_repeatability_diagnostics_optional/`: optional additional repeatability diagnostics.
- `results_overview.csv`: compact summary across the main evidence folders.

Only `01_repeatability_multiseed/` should be treated as full multi-seed repeatability evidence. The interval, peak, and SHAP folders are representative seed 45 diagnostics.
