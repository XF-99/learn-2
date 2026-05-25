# DynamicGatedStacking Reproducibility Report

This is a paper-inspired reproducible selection analysis based on test-set pointwise loss, block bootstrap R-distributions, and pairwise dominance probabilities.
It is not a full hierarchical beta-process implementation because the fitted predictors are not generative probabilistic models.

## Split: test

- Lowest mean empirical risk model: `DynamicGatedStacking`.
- `DynamicGatedStacking` mean empirical risk: 0.0667454.
- `DynamicGatedStacking` mean pairwise win probability: 0.688.
- Reproducible threshold: 0.95; trend threshold: 0.70.

- Mean P(R_DynamicGatedStacking < R_LSTM) = 0.748 across 15 scenarios: better on average, but not enough to reject reproducibly (6 scenario-level reproducible wins).
- Mean P(R_DynamicGatedStacking < R_TCN) = 0.830 across 15 scenarios: better on average, but not enough to reject reproducibly (10 scenario-level reproducible wins).
- Mean P(R_DynamicGatedStacking < R_Transformer) = 0.486 across 15 scenarios: not stably better (4 scenario-level reproducible wins).
