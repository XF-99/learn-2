# DynamicGatedStacking Reproducibility Report

This is a paper-inspired reproducible selection analysis based on test-set pointwise loss, block bootstrap R-distributions, pairwise dominance probabilities, and optional seed-level training variability.
It is not a full hierarchical beta-process implementation because the fitted predictors are not generative probabilistic models.

## Split: test

- Lowest mean empirical risk model: `DynamicGatedStacking`.
- `DynamicGatedStacking` mean empirical risk: 0.0648988.
- `DynamicGatedStacking` mean pairwise win probability: 0.701.
- Analyzed scenarios: 150; seeds: 10.
- Reproducible threshold: 0.95; trend threshold: 0.70.

- Mean P(R_DynamicGatedStacking < R_LSTM) = 0.764 across 150 scenarios: better on average, but not enough to reject reproducibly (71 scenario-level reproducible wins).
- Mean P(R_DynamicGatedStacking < R_TCN) = 0.800 across 150 scenarios: better on average, but not enough to reject reproducibly (86 scenario-level reproducible wins).
- Mean P(R_DynamicGatedStacking < R_Transformer) = 0.540 across 150 scenarios: not stably better (16 scenario-level reproducible wins).
