import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MODELS = [
    "LSTM",
    "Transformer",
    "TCN",
    "DynamicGatedStacking",
]
DEFAULT_EXCLUDE_MODELS = [
    "Persistence",
    "Stacking",
    "DynamicGatedOnly",
    "AdaptiveWeightedStacking",
]
RESULT_DIR_NAME = "reproducible_selection"


@dataclass(frozen=True)
class PairwiseProbability:
    p_ab: float
    p_ba: float
    a_win_rate: float
    b_win_rate: float
    tie_rate: float


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("yes", "true", "t", "1", "y"):
        return True
    if lowered in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_csv_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = str(value).split(",")
    return [str(item).strip() for item in items if str(item).strip()]


def selected_models(include_persistence=True, available_columns=None, exclude_models=None, model_candidates=None):
    excluded = set(DEFAULT_EXCLUDE_MODELS if exclude_models is None else parse_csv_list(exclude_models))
    if not include_persistence:
        excluded.add("Persistence")
    candidates = list(DEFAULT_MODELS if model_candidates is None else parse_csv_list(model_candidates))
    if available_columns is not None:
        available = set(available_columns)
        candidates = [model for model in candidates if model in available]
    return [model for model in candidates if model not in excluded]


def squared_loss(actual, prediction):
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return np.square(actual - prediction)


def validate_prediction_columns(frame, models):
    missing = []
    if "Actual" not in frame.columns:
        missing.append("Actual")
    missing.extend(model for model in models if model not in frame.columns)
    if missing:
        raise ValueError(f"Missing required prediction column(s): {', '.join(missing)}")


def resolve_effective_block_size(n, method, block_size, emit_warning=True):
    if n <= 0:
        raise ValueError("Cannot bootstrap an empty time series.")
    if method == "iid":
        return int(block_size)
    if method != "block":
        raise ValueError("bootstrap_method must be 'iid' or 'block'.")
    effective_block_size = int(block_size)
    if effective_block_size <= 0:
        raise ValueError("block_size must be positive.")
    if n < effective_block_size:
        message = f"block_size={effective_block_size} is larger than n={n}; using block_size={n}."
        if emit_warning:
            warnings.warn(message, UserWarning, stacklevel=2)
            print(f"Warning: {message}")
        effective_block_size = n
    return effective_block_size


def bootstrap_indices(n, bootstrap_samples, method, block_size, rng):
    if n <= 0:
        raise ValueError("Cannot bootstrap an empty time series.")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    if method == "iid":
        return rng.integers(0, n, size=(bootstrap_samples, n))
    if method != "block":
        raise ValueError("bootstrap_method must be 'iid' or 'block'.")

    effective_block_size = resolve_effective_block_size(n, method, block_size, emit_warning=True)

    samples = np.empty((bootstrap_samples, n), dtype=int)
    max_start = n - effective_block_size
    for sample_idx in range(bootstrap_samples):
        chunks = []
        while sum(len(chunk) for chunk in chunks) < n:
            start = int(rng.integers(0, max_start + 1))
            chunks.append(np.arange(start, start + effective_block_size, dtype=int))
        samples[sample_idx] = np.concatenate(chunks)[:n]
    return samples


def bootstrap_risk_distribution(losses, indices):
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1:
        raise ValueError("losses must be one-dimensional.")
    return losses[indices].mean(axis=1)


def pairwise_probability(risk_a, risk_b):
    risk_a = np.asarray(risk_a, dtype=float)
    risk_b = np.asarray(risk_b, dtype=float)
    if risk_a.shape != risk_b.shape:
        raise ValueError("risk_a and risk_b must have the same shape.")

    ties = np.isclose(risk_a, risk_b, rtol=1e-12, atol=1e-12)
    a_wins = (risk_a < risk_b) & (~ties)
    b_wins = (risk_b < risk_a) & (~ties)
    a_win_rate = float(np.mean(a_wins))
    b_win_rate = float(np.mean(b_wins))
    tie_rate = float(np.mean(ties))
    p_ab = a_win_rate + 0.5 * tie_rate
    p_ba = b_win_rate + 0.5 * tie_rate
    if abs((p_ab + p_ba) - 1.0) >= 1e-9:
        raise ValueError(
            f"Pairwise probability symmetry failed: p_ab={p_ab:.12f}, p_ba={p_ba:.12f}"
        )
    return PairwiseProbability(
        p_ab=float(p_ab),
        p_ba=float(p_ba),
        a_win_rate=a_win_rate,
        b_win_rate=b_win_rate,
        tie_rate=tie_rate,
    )


def is_stable_dominance(probability, threshold):
    return float(probability) >= float(threshold)


def dominance_label(probability, repro_threshold, trend_threshold):
    probability = float(probability)
    if probability >= float(repro_threshold):
        return "reproducible"
    if probability >= float(trend_threshold):
        return "trend"
    return "unstable"


def _split_filename(split):
    if split == "test":
        return "test_predictions.csv"
    if split == "future_holdout":
        return "future_holdout_predictions.csv"
    return f"{split}_predictions.csv"


def _well_directories(out_dir):
    out_dir = Path(out_dir)
    return sorted(
        path
        for path in out_dir.iterdir()
        if path.is_dir() and path.name != RESULT_DIR_NAME and any(path.glob("*_predictions.csv"))
    )


def _scenario_aquifer(frame):
    if "Aquifer" in frame.columns and len(frame) > 0:
        return str(frame["Aquifer"].iloc[0])
    return ""


def _load_scenarios(out_dir, splits, models=None, exclude_models=None):
    scenarios = []
    for well_dir in _well_directories(out_dir):
        for split in splits:
            prediction_path = well_dir / _split_filename(split)
            if not prediction_path.exists():
                continue
            frame = pd.read_csv(prediction_path)
            scenarios.append(
                {
                    "split": split,
                    "well": well_dir.name,
                    "aquifer": _scenario_aquifer(frame),
                    "frame": frame,
                    "path": prediction_path,
                }
            )
    if not scenarios:
        raise ValueError(f"No prediction CSV files found under {Path(out_dir)} for splits: {splits}")
    if models is None:
        common_columns = set(scenarios[0]["frame"].columns)
        for scenario in scenarios[1:]:
            common_columns &= set(scenario["frame"].columns)
        models = selected_models(available_columns=common_columns, exclude_models=exclude_models)
    if not models:
        raise ValueError("No analyzable model prediction columns were found.")
    for scenario in scenarios:
        validate_prediction_columns(scenario["frame"], models)
    return scenarios, list(models)


def _loss_quantile_rows(scenario, model, point_loss, quantile_grid_size):
    grid_size = max(int(quantile_grid_size), 2)
    levels = np.linspace(0.0, 1.0, grid_size)
    quantiles = np.quantile(np.asarray(point_loss, dtype=float), levels)
    rows = []
    for level, value in zip(levels, quantiles):
        rows.append(
            {
                "split": scenario["split"],
                "well": scenario["well"],
                "aquifer": scenario["aquifer"],
                "model": model,
                "quantile": float(level),
                "loss_quantile": float(value),
                "n_points": int(len(point_loss)),
            }
        )
    return rows


def _compute_scenario_records(
    scenario,
    models,
    loss,
    bootstrap_samples,
    bootstrap_method,
    block_size,
    dominance_threshold,
    trend_threshold,
    save_risk_samples,
    quantile_grid_size,
    rng,
):
    frame = scenario["frame"]
    actual = frame["Actual"].to_numpy(dtype=float)
    if loss != "squared":
        raise ValueError("Only squared loss is supported.")
    effective_block_size = resolve_effective_block_size(
        len(actual), bootstrap_method, block_size, emit_warning=True
    )

    indices = bootstrap_indices(
        len(actual),
        bootstrap_samples=bootstrap_samples,
        method=bootstrap_method,
        block_size=effective_block_size,
        rng=rng,
    )
    risk_by_model = {}
    risk_rows = []
    risk_sample_rows = []
    quantile_rows = []
    for model in models:
        point_loss = squared_loss(actual, frame[model].to_numpy(dtype=float))
        risk = bootstrap_risk_distribution(point_loss, indices)
        risk_by_model[model] = risk
        quantile_rows.extend(_loss_quantile_rows(scenario, model, point_loss, quantile_grid_size))
        if save_risk_samples:
            for sample_index, value in enumerate(risk):
                risk_sample_rows.append(
                    {
                        "split": scenario["split"],
                        "well": scenario["well"],
                        "aquifer": scenario["aquifer"],
                        "model": model,
                        "sample_index": int(sample_index),
                        "risk": float(value),
                    }
                )
        risk_rows.append(
            {
                "split": scenario["split"],
                "well": scenario["well"],
                "aquifer": scenario["aquifer"],
                "model": model,
                "loss": loss,
                "bootstrap_method": bootstrap_method,
                "block_size": effective_block_size,
                "mean_risk": float(np.mean(risk)),
                "risk_q05": float(np.quantile(risk, 0.05)),
                "risk_q50": float(np.quantile(risk, 0.50)),
                "risk_q95": float(np.quantile(risk, 0.95)),
                "n_points": int(len(point_loss)),
            }
        )

    pairwise_rows = []
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            result = pairwise_probability(risk_by_model[model_a], risk_by_model[model_b])
            label_ab = dominance_label(result.p_ab, dominance_threshold, trend_threshold)
            label_ba = dominance_label(result.p_ba, dominance_threshold, trend_threshold)
            pairwise_rows.append(
                {
                    "split": scenario["split"],
                    "well": scenario["well"],
                    "aquifer": scenario["aquifer"],
                    "model_a": model_a,
                    "model_b": model_b,
                    "p_a_better_than_b": result.p_ab,
                    "p_b_better_than_a": result.p_ba,
                    "a_win_rate": result.a_win_rate,
                    "b_win_rate": result.b_win_rate,
                    "tie_rate": result.tie_rate,
                    "dominates": is_stable_dominance(result.p_ab, dominance_threshold),
                    "dominance_label": label_ab,
                    "loss": loss,
                }
            )
            pairwise_rows.append(
                {
                    "split": scenario["split"],
                    "well": scenario["well"],
                    "aquifer": scenario["aquifer"],
                    "model_a": model_b,
                    "model_b": model_a,
                    "p_a_better_than_b": result.p_ba,
                    "p_b_better_than_a": result.p_ab,
                    "a_win_rate": result.b_win_rate,
                    "b_win_rate": result.a_win_rate,
                    "tie_rate": result.tie_rate,
                    "dominates": is_stable_dominance(result.p_ba, dominance_threshold),
                    "dominance_label": label_ba,
                    "loss": loss,
                }
            )
    return risk_rows, pairwise_rows, risk_sample_rows, quantile_rows


def build_stable_rejections(pairwise, dominance_threshold):
    columns = [
        "split",
        "well",
        "aquifer",
        "rejected_model",
        "dominating_model",
        "dominance_probability",
        "reason",
    ]
    if pairwise.empty:
        return pd.DataFrame(columns=columns)
    dominated = pairwise[pairwise["p_a_better_than_b"] >= dominance_threshold].copy()
    rows = []
    for _, row in dominated.iterrows():
        probability = float(row["p_a_better_than_b"])
        rows.append(
            {
                "split": row["split"],
                "well": row["well"],
                "aquifer": row.get("aquifer", ""),
                "rejected_model": row["model_b"],
                "dominating_model": row["model_a"],
                "dominance_probability": probability,
                "reason": (
                    f"P(R_{row['model_a']} < R_{row['model_b']}) = "
                    f"{probability:.3f} >= {dominance_threshold:.2f}"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_pairwise_summary(pairwise, repro_threshold, trend_threshold):
    columns = [
        "split",
        "model_a",
        "model_b",
        "mean_p_a_better_than_b",
        "mean_p_b_better_than_a",
        "mean_tie_rate",
        "scenario_count",
        "reproducible_count",
        "trend_count",
        "dominance_label",
    ]
    if pairwise.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    grouped = pairwise.groupby(["split", "model_a", "model_b"], sort=True)
    for (split, model_a, model_b), group in grouped:
        mean_p = float(group["p_a_better_than_b"].mean())
        rows.append(
            {
                "split": split,
                "model_a": model_a,
                "model_b": model_b,
                "mean_p_a_better_than_b": mean_p,
                "mean_p_b_better_than_a": float(group["p_b_better_than_a"].mean()),
                "mean_tie_rate": float(group["tie_rate"].mean()),
                "scenario_count": int(len(group)),
                "reproducible_count": int((group["dominance_label"] == "reproducible").sum()),
                "trend_count": int((group["dominance_label"] == "trend").sum()),
                "dominance_label": dominance_label(mean_p, repro_threshold, trend_threshold),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _ranking_block_size(model_risk, fallback_block_size):
    if model_risk.empty or "block_size" not in model_risk.columns:
        return fallback_block_size
    values = sorted(model_risk["block_size"].dropna().unique())
    if len(values) == 1:
        value = values[0]
        if float(value).is_integer():
            return int(value)
        return value
    return "mixed:" + "|".join(str(value) for value in values)


def build_model_ranking(risk_summary, pairwise, models, loss, bootstrap_method, block_size):
    columns = [
        "split",
        "model",
        "loss",
        "bootstrap_method",
        "block_size",
        "n_wells",
        "mean_risk",
        "mean_win_probability",
        "dominance_count",
        "rejected_count",
        "non_rejected_count",
        "stable_win_count",
        "stable_loss_count",
        "trend_win_count",
        "trend_loss_count",
    ]
    if risk_summary.empty:
        return pd.DataFrame(columns=columns)

    ranking_rows = []
    for split in sorted(risk_summary["split"].unique()):
        split_risk = risk_summary[risk_summary["split"] == split]
        split_pairwise = pairwise[pairwise["split"] == split] if not pairwise.empty else pairwise
        wells = sorted(split_risk["well"].unique())
        for model in models:
            model_risk = split_risk[split_risk["model"] == model]
            as_a = split_pairwise[split_pairwise["model_a"] == model] if not split_pairwise.empty else split_pairwise
            stable_wins = split_pairwise[
                (split_pairwise["model_a"] == model) & (split_pairwise["dominates"])
            ]
            stable_losses = split_pairwise[
                (split_pairwise["model_b"] == model) & (split_pairwise["dominates"])
            ]
            if "dominance_label" in split_pairwise.columns:
                trend_wins = split_pairwise[
                    (split_pairwise["model_a"] == model) & (split_pairwise["dominance_label"] == "trend")
                ]
                trend_losses = split_pairwise[
                    (split_pairwise["model_b"] == model) & (split_pairwise["dominance_label"] == "trend")
                ]
            else:
                trend_wins = split_pairwise.iloc[0:0]
                trend_losses = split_pairwise.iloc[0:0]
            rejected_scenarios = {
                (row["split"], row["well"]) for _, row in stable_losses.iterrows()
            }
            possible_scenarios = {
                (row["split"], row["well"]) for _, row in model_risk.iterrows()
            }
            rejected_count = len(rejected_scenarios)
            non_rejected_count = len(possible_scenarios - rejected_scenarios)
            stable_win_count = int(len(stable_wins))
            stable_loss_count = int(len(stable_losses))
            ranking_rows.append(
                {
                    "split": split,
                    "model": model,
                    "loss": loss,
                    "bootstrap_method": bootstrap_method,
                    "block_size": _ranking_block_size(model_risk, block_size),
                    "n_wells": len(wells),
                    "mean_risk": float(model_risk["mean_risk"].mean()) if not model_risk.empty else np.nan,
                    "mean_win_probability": (
                        float(as_a["p_a_better_than_b"].mean()) if not as_a.empty else np.nan
                    ),
                    "dominance_count": stable_win_count + stable_loss_count,
                    "rejected_count": rejected_count,
                    "non_rejected_count": non_rejected_count,
                    "stable_win_count": stable_win_count,
                    "stable_loss_count": stable_loss_count,
                    "trend_win_count": int(len(trend_wins)),
                    "trend_loss_count": int(len(trend_losses)),
                }
            )
    return pd.DataFrame(ranking_rows, columns=columns)


def _probability_matrix(pairwise, models):
    matrix = pd.DataFrame(np.eye(len(models)) * 0.5, index=models, columns=models)
    for model_a in models:
        for model_b in models:
            if model_a == model_b:
                continue
            rows = pairwise[(pairwise["model_a"] == model_a) & (pairwise["model_b"] == model_b)]
            if not rows.empty:
                matrix.loc[model_a, model_b] = float(rows["p_a_better_than_b"].mean())
    return matrix


def _save_heatmap(matrix, path, title):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(matrix.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_title(title)
    for row_idx, row_name in enumerate(matrix.index):
        for col_idx, col_name in enumerate(matrix.columns):
            value = matrix.loc[row_name, col_name]
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(R_A < R_B)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _safe_label(value):
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return label or "well"


def _prepare_result_dir(result_dir):
    result_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "loss_quantile_functions.csv",
        "risk_distribution_samples.csv",
        "risk_distribution_summary.csv",
        "pairwise_reproducible_dominance.csv",
        "pairwise_reproducible_dominance_summary.csv",
        "pairwise_dominance_probabilities.csv",
        "stable_model_rejections.csv",
        "model_reproducibility_ranking.csv",
        "dynamic_gated_reproducibility_report.md",
    ):
        path = result_dir / filename
        if path.exists():
            path.unlink()
    for path in result_dir.glob("pairwise_probability_heatmap_*.png"):
        path.unlink()
    for path in result_dir.glob("risk_distribution_plot_*.png"):
        path.unlink()
    per_well_dir = result_dir / "per_well_heatmaps"
    if per_well_dir.exists():
        for path in per_well_dir.glob("pairwise_probability_heatmap_*.png"):
            path.unlink()
    per_well_dir.mkdir(parents=True, exist_ok=True)


def write_heatmaps(pairwise, models, result_dir):
    per_well_dir = result_dir / "per_well_heatmaps"
    per_well_dir.mkdir(parents=True, exist_ok=True)
    for split in sorted(pairwise["split"].unique()):
        split_pairwise = pairwise[pairwise["split"] == split]
        matrix = _probability_matrix(split_pairwise, models)
        _save_heatmap(
            matrix,
            result_dir / f"pairwise_probability_heatmap_{split}.png",
            f"{split} mean pairwise probability",
        )
        for well_index, well in enumerate(sorted(split_pairwise["well"].unique()), start=1):
            well_pairwise = split_pairwise[split_pairwise["well"] == well]
            well_matrix = _probability_matrix(well_pairwise, models)
            safe_well = f"{well_index:02d}_{_safe_label(well)}"
            _save_heatmap(
                well_matrix,
                per_well_dir / f"pairwise_probability_heatmap_{split}_{safe_well}.png",
                f"{split} {safe_well}",
            )


def write_risk_distribution_plots(risk_samples, models, result_dir):
    if risk_samples.empty:
        return
    for split in sorted(risk_samples["split"].unique()):
        split_samples = risk_samples[risk_samples["split"] == split]
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in models:
            values = split_samples.loc[split_samples["model"] == model, "risk"].to_numpy(dtype=float)
            if len(values) == 0:
                continue
            if np.isclose(np.nanmin(values), np.nanmax(values)):
                ax.axvline(float(values[0]), alpha=0.8, label=model)
            else:
                ax.hist(values, bins=min(40, max(1, len(values) // 2)), density=True, alpha=0.35, label=model)
        ax.set_title(f"{split} R-distribution")
        ax.set_xlabel("Empirical risk")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(result_dir / f"risk_distribution_plot_{split}.png", dpi=160)
        plt.close(fig)


def write_dynamic_gated_report(ranking, pairwise_summary, target_model, result_dir, repro_threshold, trend_threshold):
    report_path = result_dir / "dynamic_gated_reproducibility_report.md"
    lines = [
        "# DynamicGatedStacking Reproducibility Report",
        "",
        "This is a paper-inspired reproducible selection analysis based on test-set pointwise loss, block bootstrap R-distributions, and pairwise dominance probabilities.",
        "It is not a full hierarchical beta-process implementation because the fitted predictors are not generative probabilistic models.",
        "",
    ]
    if ranking.empty or target_model not in set(ranking["model"]):
        lines.extend([f"Target model `{target_model}` was not found in the analyzed prediction columns."])
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    for split in sorted(ranking["split"].unique()):
        split_ranking = ranking[ranking["split"] == split].sort_values("mean_risk")
        target_rows = split_ranking[split_ranking["model"] == target_model]
        if target_rows.empty:
            continue
        target_row = target_rows.iloc[0]
        best_model = str(split_ranking.iloc[0]["model"])
        lines.extend(
            [
                f"## Split: {split}",
                "",
                f"- Lowest mean empirical risk model: `{best_model}`.",
                f"- `{target_model}` mean empirical risk: {float(target_row['mean_risk']):.6g}.",
                f"- `{target_model}` mean pairwise win probability: {float(target_row['mean_win_probability']):.3f}.",
                f"- Reproducible threshold: {float(repro_threshold):.2f}; trend threshold: {float(trend_threshold):.2f}.",
                "",
            ]
        )
        split_pairwise = pairwise_summary[
            (pairwise_summary["split"] == split) & (pairwise_summary["model_a"] == target_model)
        ]
        if split_pairwise.empty:
            lines.append(f"- No pairwise comparisons were available for `{target_model}`.")
            lines.append("")
            continue
        for _, row in split_pairwise.sort_values("model_b").iterrows():
            p = float(row["mean_p_a_better_than_b"])
            other = row["model_b"]
            label = row.get("dominance_label", dominance_label(p, repro_threshold, trend_threshold))
            if label == "reproducible":
                wording = "reproducibly better"
            elif label == "trend":
                wording = "better on average, but not enough to reject reproducibly"
            else:
                wording = "not stably better"
            lines.append(
                f"- Mean P(R_{target_model} < R_{other}) = {p:.3f} "
                f"across {int(row['scenario_count'])} scenarios: {wording} "
                f"({int(row['reproducible_count'])} scenario-level reproducible wins)."
            )
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    out_dir,
    splits,
    loss,
    bootstrap_samples,
    bootstrap_method,
    block_size,
    seed=42,
    dominance_threshold=0.95,
    include_persistence=False,
    trend_threshold=0.70,
    exclude_models=None,
    target_model="DynamicGatedStacking",
    save_risk_samples=True,
    quantile_grid_size=101,
    models=None,
):
    out_dir = Path(out_dir)
    exclude_models = DEFAULT_EXCLUDE_MODELS if exclude_models is None else parse_csv_list(exclude_models)
    if models is not None:
        models = selected_models(
            include_persistence=include_persistence,
            available_columns=models,
            exclude_models=exclude_models,
            model_candidates=models,
        )
    scenarios, models = _load_scenarios(out_dir, splits, models=models, exclude_models=exclude_models)
    rng = np.random.default_rng(seed)

    risk_rows = []
    pairwise_rows = []
    risk_sample_rows = []
    quantile_rows = []
    for scenario in scenarios:
        scenario_risk, scenario_pairwise, scenario_samples, scenario_quantiles = _compute_scenario_records(
            scenario=scenario,
            models=models,
            loss=loss,
            bootstrap_samples=bootstrap_samples,
            bootstrap_method=bootstrap_method,
            block_size=block_size,
            dominance_threshold=dominance_threshold,
            trend_threshold=trend_threshold,
            save_risk_samples=save_risk_samples,
            quantile_grid_size=quantile_grid_size,
            rng=rng,
        )
        risk_rows.extend(scenario_risk)
        pairwise_rows.extend(scenario_pairwise)
        risk_sample_rows.extend(scenario_samples)
        quantile_rows.extend(scenario_quantiles)

    risk_summary = pd.DataFrame(risk_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    risk_samples = pd.DataFrame(risk_sample_rows)
    loss_quantiles = pd.DataFrame(quantile_rows)
    rejections = build_stable_rejections(pairwise, dominance_threshold)
    pairwise_summary = build_pairwise_summary(pairwise, dominance_threshold, trend_threshold)
    ranking = build_model_ranking(
        risk_summary=risk_summary,
        pairwise=pairwise,
        models=models,
        loss=loss,
        bootstrap_method=bootstrap_method,
        block_size=block_size,
    )

    result_dir = out_dir / RESULT_DIR_NAME
    _prepare_result_dir(result_dir)
    loss_quantiles.to_csv(result_dir / "loss_quantile_functions.csv", index=False)
    if save_risk_samples:
        risk_samples.to_csv(result_dir / "risk_distribution_samples.csv", index=False)
    risk_summary.to_csv(result_dir / "risk_distribution_summary.csv", index=False)
    pairwise.to_csv(result_dir / "pairwise_reproducible_dominance.csv", index=False)
    pairwise_summary.to_csv(result_dir / "pairwise_reproducible_dominance_summary.csv", index=False)
    pairwise.to_csv(result_dir / "pairwise_dominance_probabilities.csv", index=False)
    rejections.to_csv(result_dir / "stable_model_rejections.csv", index=False)
    ranking.to_csv(result_dir / "model_reproducibility_ranking.csv", index=False)
    write_heatmaps(pairwise, models, result_dir)
    write_risk_distribution_plots(risk_samples, models, result_dir)
    write_dynamic_gated_report(
        ranking,
        pairwise_summary,
        target_model,
        result_dir,
        dominance_threshold,
        trend_threshold,
    )

    return {
        "risk_summary": risk_summary,
        "risk_samples": risk_samples,
        "loss_quantiles": loss_quantiles,
        "pairwise": pairwise,
        "pairwise_summary": pairwise_summary,
        "rejections": rejections,
        "ranking": ranking,
        "result_dir": result_dir,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="R-distribution reproducible model selection analysis."
    )
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--splits", default="test")
    parser.add_argument("--loss", default="squared", choices=["squared"])
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_method", default="block", choices=["iid", "block"])
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--dominance_threshold", type=float, default=0.95)
    parser.add_argument("--repro_threshold", type=float, default=None)
    parser.add_argument("--trend_threshold", type=float, default=0.70)
    parser.add_argument("--include_persistence", type=str2bool, default=False)
    parser.add_argument("--exclude_models", default=",".join(DEFAULT_EXCLUDE_MODELS))
    parser.add_argument("--target_model", default="DynamicGatedStacking")
    parser.add_argument("--save_risk_samples", type=str2bool, default=True)
    parser.add_argument("--quantile_grid_size", type=int, default=101)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    dominance_threshold = args.dominance_threshold
    if args.repro_threshold is not None:
        dominance_threshold = args.repro_threshold
    result = run_analysis(
        out_dir=args.out_dir,
        splits=splits,
        loss=args.loss,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_method=args.bootstrap_method,
        block_size=args.block_size,
        dominance_threshold=dominance_threshold,
        include_persistence=args.include_persistence,
        seed=args.seed,
        trend_threshold=args.trend_threshold,
        exclude_models=parse_csv_list(args.exclude_models),
        target_model=args.target_model,
        save_risk_samples=args.save_risk_samples,
        quantile_grid_size=args.quantile_grid_size,
    )
    print(f"Wrote reproducible model selection outputs to {result['result_dir']}")


if __name__ == "__main__":
    main()
