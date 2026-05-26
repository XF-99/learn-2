# -*- coding: utf-8 -*-
"""EMD-inspired reproducible model selection for fitted deterministic predictors.

This adapts the EMD/BEMD paper's quantile-function idea to non-generative
groundwater predictors by using residual block replicates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MODELS = ["LSTM", "Transformer", "TCN", "DynamicGatedStacking"]
RESULT_DIR_NAME = "emd_reproducible_selection"
PREDICTION_FILE = "multiseed_test_predictions_long.csv"


def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def squared_loss(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.square(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))


def quantile_function(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty one-dimensional array.")
    return np.quantile(values, levels)


def block_resample(values: np.ndarray, n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("Cannot resample an empty residual array.")
    effective_block = min(max(int(block_size), 1), len(values))
    max_start = len(values) - effective_block
    chunks = []
    while sum(len(chunk) for chunk in chunks) < n:
        start = int(rng.integers(0, max_start + 1))
        chunks.append(values[start : start + effective_block])
    return np.concatenate(chunks)[:n]


def pairwise_probability(risk_a: np.ndarray, risk_b: np.ndarray) -> tuple[float, float]:
    risk_a = np.asarray(risk_a, dtype=float)
    risk_b = np.asarray(risk_b, dtype=float)
    if risk_a.shape != risk_b.shape:
        raise ValueError("risk arrays must have the same shape.")
    ties = np.isclose(risk_a, risk_b, rtol=1e-12, atol=1e-12)
    a_wins = (risk_a < risk_b) & (~ties)
    tie_rate = float(np.mean(ties))
    return float(np.mean(a_wins) + 0.5 * tie_rate), tie_rate


def dominance_label(probability: float, repro_threshold: float, trend_threshold: float) -> str:
    if probability >= repro_threshold:
        return "reproducible"
    if probability >= trend_threshold:
        return "trend"
    return "unstable"


def load_predictions(out_dir: Path, models: list[str]) -> pd.DataFrame:
    path = out_dir / PREDICTION_FILE
    if not path.exists():
        raise FileNotFoundError(f"Expected multiseed prediction file: {path}")
    frame = pd.read_csv(path)
    required = {"seed", "well", "Actual", *models}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
    return frame


def scenario_groups(frame: pd.DataFrame):
    return frame.groupby(["seed", "well"], sort=True)


def build_seed_truth(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for (seed, well), group in scenario_groups(frame):
        actual = group["Actual"].to_numpy(dtype=float)
        risks = {model: float(np.mean(squared_loss(actual, group[model].to_numpy(dtype=float)))) for model in models}
        for model_a in models:
            for model_b in models:
                if model_a == model_b:
                    continue
                rows.append(
                    {
                        "seed": int(seed),
                        "well": str(well),
                        "model_a": model_a,
                        "model_b": model_b,
                        "a_better": float(risks[model_a] < risks[model_b]),
                    }
                )
    return pd.DataFrame(rows)


def compute_model_samples(
    group: pd.DataFrame,
    model: str,
    levels: np.ndarray,
    c_values: np.ndarray,
    n_samples: int,
    block_size: int,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], dict[float, np.ndarray]]:
    actual = group["Actual"].to_numpy(dtype=float)
    pred = group[model].to_numpy(dtype=float)
    residual = actual - pred
    base_loss = squared_loss(actual, pred)
    q_emp = quantile_function(base_loss, levels)

    q_rep_samples = []
    for _ in range(n_samples):
        sampled_residual = block_resample(residual, len(residual), block_size, rng)
        rep_actual = pred + sampled_residual
        rep_loss = squared_loss(rep_actual, pred)
        q_rep_samples.append(quantile_function(rep_loss, levels))
    q_rep = np.vstack(q_rep_samples)
    q_rep_mean = q_rep.mean(axis=0)
    delta = np.abs(q_rep_mean - q_emp)
    empirical_risk = float(np.trapezoid(q_emp, levels))

    quantile_rows = []
    discrepancy_rows = []
    for level, empirical, replicate, discrepancy in zip(levels, q_emp, q_rep_mean, delta):
        quantile_rows.append(
            {
                "model": model,
                "quantile": float(level),
                "empirical_loss_quantile": float(empirical),
                "replicate_loss_quantile": float(replicate),
            }
        )
        discrepancy_rows.append(
            {
                "model": model,
                "quantile": float(level),
                "delta_emd": float(discrepancy),
            }
        )

    risk_by_c: dict[float, np.ndarray] = {}
    centered = q_rep - q_rep_mean
    random_sign = rng.choice(np.array([-1.0, 1.0]), size=(n_samples, len(levels)))
    for c in c_values:
        q_samples = q_emp + centered + random_sign * float(c) * delta
        q_samples = np.maximum(q_samples, 0.0)
        q_samples.sort(axis=1)
        risk_by_c[float(c)] = np.trapezoid(q_samples, levels, axis=1)

    summary = {
        "model": model,
        "empirical_risk": empirical_risk,
        "mean_delta_emd": float(np.mean(delta)),
        "max_delta_emd": float(np.max(delta)),
    }
    for row in quantile_rows:
        row.update(summary)
    for row in discrepancy_rows:
        row.update(summary)
    return quantile_rows, discrepancy_rows, risk_by_c


def run_analysis(
    out_dir: Path,
    models: list[str],
    c_values: list[float],
    n_samples: int,
    block_size: int,
    quantile_grid_size: int,
    seed: int,
    repro_threshold: float,
    trend_threshold: float,
) -> dict[str, pd.DataFrame | Path]:
    frame = load_predictions(out_dir, models)
    result_dir = out_dir / RESULT_DIR_NAME
    result_dir.mkdir(parents=True, exist_ok=True)

    levels = np.linspace(0.0, 1.0, max(int(quantile_grid_size), 2))
    c_array = np.asarray(c_values, dtype=float)
    rng = np.random.default_rng(seed)
    truth = build_seed_truth(frame, models)

    quantile_rows = []
    discrepancy_rows = []
    risk_rows = []
    pairwise_rows = []
    calibration_rows = []

    for (scenario_seed, well), group in scenario_groups(frame):
        risk_by_model_and_c: dict[str, dict[float, np.ndarray]] = {}
        for model in models:
            q_rows, d_rows, risk_by_c = compute_model_samples(
                group=group,
                model=model,
                levels=levels,
                c_values=c_array,
                n_samples=n_samples,
                block_size=block_size,
                rng=rng,
            )
            for row in q_rows:
                row.update({"seed": int(scenario_seed), "well": str(well), "scenario_id": f"seed_{int(scenario_seed)}:{well}"})
            for row in d_rows:
                row.update({"seed": int(scenario_seed), "well": str(well), "scenario_id": f"seed_{int(scenario_seed)}:{well}"})
            quantile_rows.extend(q_rows)
            discrepancy_rows.extend(d_rows)
            risk_by_model_and_c[model] = risk_by_c
            for c, risks in risk_by_c.items():
                risk_rows.append(
                    {
                        "seed": int(scenario_seed),
                        "well": str(well),
                        "scenario_id": f"seed_{int(scenario_seed)}:{well}",
                        "model": model,
                        "c": float(c),
                        "risk_mean": float(np.mean(risks)),
                        "risk_q05": float(np.quantile(risks, 0.05)),
                        "risk_q50": float(np.quantile(risks, 0.50)),
                        "risk_q95": float(np.quantile(risks, 0.95)),
                    }
                )

        for c in c_array:
            for model_a in models:
                for model_b in models:
                    if model_a == model_b:
                        continue
                    probability, tie_rate = pairwise_probability(
                        risk_by_model_and_c[model_a][float(c)],
                        risk_by_model_and_c[model_b][float(c)],
                    )
                    pairwise_rows.append(
                        {
                            "seed": int(scenario_seed),
                            "well": str(well),
                            "scenario_id": f"seed_{int(scenario_seed)}:{well}",
                            "model_a": model_a,
                            "model_b": model_b,
                            "c": float(c),
                            "p_a_better_than_b": probability,
                            "tie_rate": tie_rate,
                            "dominance_label": dominance_label(probability, repro_threshold, trend_threshold),
                        }
                    )

    pairwise = pd.DataFrame(pairwise_rows)
    truth_by_pair = (
        truth.groupby(["well", "model_a", "model_b"], as_index=False)["a_better"]
        .mean()
        .rename(columns={"a_better": "seed_observed_probability"})
    )
    calibration_detail_rows = []
    for c, group in pairwise.groupby("c", sort=True):
        predicted_by_pair = (
            group.groupby(["well", "model_a", "model_b"], as_index=False)["p_a_better_than_b"]
            .mean()
            .rename(columns={"p_a_better_than_b": "predicted_probability"})
        )
        merged = predicted_by_pair.merge(truth_by_pair, on=["well", "model_a", "model_b"], how="left")
        merged["absolute_calibration_error"] = np.abs(
            merged["predicted_probability"] - merged["seed_observed_probability"]
        )
        for _, row in merged.iterrows():
            calibration_detail_rows.append(
                {
                    "c": float(c),
                    "well": row["well"],
                    "model_a": row["model_a"],
                    "model_b": row["model_b"],
                    "predicted_probability": float(row["predicted_probability"]),
                    "seed_observed_probability": float(row["seed_observed_probability"]),
                    "absolute_calibration_error": float(row["absolute_calibration_error"]),
                }
            )
        calibration_rows.append(
            {
                "c": float(c),
                "mean_pair_predicted_probability": float(merged["predicted_probability"].mean()),
                "mean_pair_seed_observed_probability": float(merged["seed_observed_probability"].mean()),
                "mean_absolute_pair_calibration_error": float(merged["absolute_calibration_error"].mean()),
            }
        )

    calibration_detail = pd.DataFrame(calibration_detail_rows)
    calibration = pd.DataFrame(calibration_rows).sort_values("mean_absolute_pair_calibration_error")
    selected_c = float(calibration.iloc[0]["c"])
    selected = pairwise[pairwise["c"].eq(selected_c)]
    summary_rows = []
    for (model_a, model_b), group in selected.groupby(["model_a", "model_b"], sort=True):
        mean_p = float(group["p_a_better_than_b"].mean())
        summary_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "selected_c": selected_c,
                "mean_p_a_better_than_b": mean_p,
                "scenario_count": int(len(group)),
                "reproducible_count": int((group["dominance_label"] == "reproducible").sum()),
                "trend_count": int((group["dominance_label"] == "trend").sum()),
                "dominance_label": dominance_label(mean_p, repro_threshold, trend_threshold),
            }
        )
    summary = pd.DataFrame(summary_rows)

    quantiles = pd.DataFrame(quantile_rows)
    discrepancies = pd.DataFrame(discrepancy_rows)
    risks = pd.DataFrame(risk_rows)
    quantiles.to_csv(result_dir / "emd_loss_quantile_functions.csv", index=False)
    discrepancies.to_csv(result_dir / "emd_discrepancy_functions.csv", index=False)
    risks.to_csv(result_dir / "emd_risk_distribution_summary.csv", index=False)
    pairwise.to_csv(result_dir / "emd_pairwise_probabilities_by_c.csv", index=False)
    calibration_detail.to_csv(result_dir / "emd_c_calibration_detail.csv", index=False)
    calibration.to_csv(result_dir / "emd_c_calibration.csv", index=False)
    summary.to_csv(result_dir / "emd_pairwise_summary_selected_c.csv", index=False)
    write_plots(calibration, summary, result_dir)
    write_report(summary, calibration, result_dir, repro_threshold, trend_threshold)
    return {
        "result_dir": result_dir,
        "calibration": calibration,
        "summary": summary,
    }


def write_plots(calibration: pd.DataFrame, summary: pd.DataFrame, result_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(calibration["c"], calibration["mean_absolute_pair_calibration_error"], marker="o")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("c")
    ax.set_ylabel("Mean absolute calibration error")
    ax.set_title("EMD-lite c calibration")
    fig.tight_layout()
    fig.savefig(result_dir / "emd_c_calibration_plot.png", dpi=160)
    plt.close(fig)

    models = sorted(set(summary["model_a"]) | set(summary["model_b"]))
    matrix = pd.DataFrame(np.eye(len(models)) * 0.5, index=models, columns=models)
    for _, row in summary.iterrows():
        matrix.loc[row["model_a"], row["model_b"]] = float(row["mean_p_a_better_than_b"])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(models)), labels=models, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(models)), labels=models)
    ax.set_title("Selected-c EMD-lite pairwise probability")
    for i, row_name in enumerate(models):
        for j, col_name in enumerate(models):
            ax.text(j, i, f"{matrix.loc[row_name, col_name]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="P(R_A < R_B)")
    fig.tight_layout()
    fig.savefig(result_dir / "emd_pairwise_probability_heatmap_selected_c.png", dpi=160)
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    calibration: pd.DataFrame,
    result_dir: Path,
    repro_threshold: float,
    trend_threshold: float,
) -> None:
    selected_c = float(calibration.iloc[0]["c"])
    target = "DynamicGatedStacking"
    lines = [
        "# EMD-lite Reproducible Model Selection Report",
        "",
        "This analysis adapts the EMD/BEMD paper to deterministic groundwater predictors using residual block replicates.",
        "It uses loss quantile functions, residual-replicate discrepancy functions, selected-c R-distributions, and pairwise probabilities.",
        "It is still not the paper's full generative hierarchical beta-process BEMD implementation.",
        "",
        f"- Selected c: {selected_c:g}",
        f"- Pair-level calibration MAE: {float(calibration.iloc[0]['mean_absolute_pair_calibration_error']):.4f}",
        f"- Reproducible threshold: {repro_threshold:.2f}; trend threshold: {trend_threshold:.2f}.",
        "",
        f"## Target: {target}",
        "",
    ]
    target_rows = summary[summary["model_a"].eq(target)].sort_values("model_b")
    for _, row in target_rows.iterrows():
        p = float(row["mean_p_a_better_than_b"])
        label = dominance_label(p, repro_threshold, trend_threshold)
        if label == "reproducible":
            wording = "reproducibly better"
        elif label == "trend":
            wording = "better on average, but not enough to reject reproducibly"
        else:
            wording = "not stably better"
        lines.append(
            f"- Mean P(R_{target} < R_{row['model_b']} | c={selected_c:g}) = {p:.3f} "
            f"across {int(row['scenario_count'])} scenarios: {wording} "
            f"({int(row['reproducible_count'])} scenario-level reproducible wins)."
        )
    lines.append("")
    lines.append("## Calibration Grid")
    lines.append("")
    for _, row in calibration.sort_values("c").iterrows():
        lines.append(
            f"- c={float(row['c']):g}: pair calibration MAE={float(row['mean_absolute_pair_calibration_error']):.4f}, "
            f"predicted pair mean={float(row['mean_pair_predicted_probability']):.3f}, "
            f"observed pair mean={float(row['mean_pair_seed_observed_probability']):.3f}"
        )
    (result_dir / "emd_lite_reproducibility_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs_15wells_multiseed_test_focus"))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--c_values", default="0,0.25,0.5,1,2,4")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--quantile_grid_size", type=int, default=101)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repro_threshold", type=float, default=0.95)
    parser.add_argument("--trend_threshold", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = parse_csv_list(args.models)
    c_values = [float(value) for value in parse_csv_list(args.c_values)]
    result = run_analysis(
        out_dir=args.out_dir,
        models=models,
        c_values=c_values,
        n_samples=args.samples,
        block_size=args.block_size,
        quantile_grid_size=args.quantile_grid_size,
        seed=args.seed,
        repro_threshold=args.repro_threshold,
        trend_threshold=args.trend_threshold,
    )
    print(f"Wrote EMD-lite reproducible model selection outputs to {result['result_dir']}")


if __name__ == "__main__":
    main()
