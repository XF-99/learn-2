# EMD-lite Reproducible Model Selection Report

This analysis adapts the EMD/BEMD paper to deterministic groundwater predictors using residual block replicates.
It uses loss quantile functions, residual-replicate discrepancy functions, selected-c R-distributions, and pairwise probabilities.
It is still not the paper's full generative hierarchical beta-process BEMD implementation.

- Selected c: 0
- Pair-level calibration MAE: 0.1849
- Reproducible threshold: 0.95; trend threshold: 0.70.

## Target: DynamicGatedStacking

- Mean P(R_DynamicGatedStacking < R_LSTM | c=0) = 0.600 across 150 scenarios: not stably better (2 scenario-level reproducible wins).
- Mean P(R_DynamicGatedStacking < R_TCN | c=0) = 0.685 across 150 scenarios: not stably better (11 scenario-level reproducible wins).
- Mean P(R_DynamicGatedStacking < R_Transformer | c=0) = 0.528 across 150 scenarios: not stably better (1 scenario-level reproducible wins).

## Calibration Grid

- c=0: pair calibration MAE=0.1849, predicted pair mean=0.500, observed pair mean=0.500
- c=0.25: pair calibration MAE=0.1850, predicted pair mean=0.500, observed pair mean=0.500
- c=0.5: pair calibration MAE=0.1856, predicted pair mean=0.500, observed pair mean=0.500
- c=1: pair calibration MAE=0.1876, predicted pair mean=0.500, observed pair mean=0.500
- c=2: pair calibration MAE=0.1911, predicted pair mean=0.500, observed pair mean=0.500
- c=4: pair calibration MAE=0.1975, predicted pair mean=0.500, observed pair mean=0.500