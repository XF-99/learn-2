# DynamicGatedStacking Reproducibility Report

This is a paper-inspired reproducible selection analysis based on test-set pointwise loss, block bootstrap R-distributions, pairwise dominance probabilities, and optional seed-level training variability.
It is not a full hierarchical beta-process implementation because the fitted predictors are not generative probabilistic models.

## Split: test

- Lowest mean empirical risk model: `DynamicGatedStacking`.
- `DynamicGatedStacking` mean empirical risk: 0.108968.
- `DynamicGatedStacking` mean pairwise win probability: 0.739.
- Analyzed scenarios: 15; seeds: 0.
- Reproducible threshold: 0.95; trend threshold: 0.70.

- Mean P(R_DynamicGatedStacking < R_LSTM) = 0.719 across 15 scenarios: better on average, but not enough to reject reproducibly (6 scenario-level reproducible wins).
- Mean P(R_DynamicGatedStacking < R_TCN) = 0.908 across 15 scenarios: better on average, but not enough to reject reproducibly (12 scenario-level reproducible wins).
- Mean P(R_DynamicGatedStacking < R_Transformer) = 0.591 across 15 scenarios: not stably better (6 scenario-level reproducible wins).
