import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import reproducible_model_selection as rms


class ReproducibleModelSelectionTests(unittest.TestCase):
    def assert_blocks_are_contiguous(self, sample, block_size):
        for start in range(0, len(sample), block_size):
            block = sample[start : start + block_size]
            if len(block) > 1:
                np.testing.assert_array_equal(np.diff(block), np.ones(len(block) - 1, dtype=int))

    def test_squared_loss_computes_pointwise_error(self):
        actual = np.array([1.0, 2.0, 4.0])
        prediction = np.array([1.5, 1.0, 6.0])

        np.testing.assert_allclose(rms.squared_loss(actual, prediction), [0.25, 1.0, 4.0])

    def test_bootstrap_indices_have_expected_shapes(self):
        rng = np.random.default_rng(123)

        iid = rms.bootstrap_indices(5, 7, method="iid", block_size=3, rng=rng)
        block = rms.bootstrap_indices(5, 7, method="block", block_size=3, rng=rng)

        self.assertEqual(iid.shape, (7, 5))
        self.assertEqual(block.shape, (7, 5))

    def test_block_bootstrap_uses_contiguous_blocks_and_exact_length(self):
        rng = np.random.default_rng(7)

        samples = rms.bootstrap_indices(10, 20, method="block", block_size=4, rng=rng)

        self.assertEqual(samples.shape, (20, 10))
        for sample in samples:
            self.assertEqual(len(sample), 10)
            self.assertTrue(np.all((sample >= 0) & (sample < 10)))
            self.assert_blocks_are_contiguous(sample, block_size=4)

    def test_block_bootstrap_reduces_oversized_block_and_warns(self):
        rng = np.random.default_rng(9)

        with self.assertWarnsRegex(UserWarning, "block_size"):
            indices = rms.bootstrap_indices(3, 2, method="block", block_size=8, rng=rng)

        self.assertEqual(indices.shape, (2, 3))

    def test_pairwise_probability_rewards_clear_winner(self):
        risk_a = np.array([1.0, 1.0, 1.0, 1.0])
        risk_b = np.array([2.0, 3.0, 2.0, 4.0])

        result = rms.pairwise_probability(risk_a, risk_b)

        self.assertEqual(result.p_ab, 1.0)
        self.assertEqual(result.p_ba, 0.0)
        self.assertTrue(rms.is_stable_dominance(result.p_ab, 0.95))

    def test_identical_models_have_half_probability_and_are_not_stable(self):
        risk = np.array([1.0, 2.0, 3.0, 4.0])

        result = rms.pairwise_probability(risk, risk.copy())

        self.assertEqual(result.p_ab, 0.5)
        self.assertEqual(result.p_ba, 0.5)
        self.assertFalse(rms.is_stable_dominance(result.p_ab, 0.95))

    def test_ties_do_not_count_as_wins_and_probabilities_are_symmetric(self):
        risk_a = np.array([1.0, 1.0, 2.0, 4.0])
        risk_b = np.array([1.0, 2.0, 2.0, 3.0])

        result = rms.pairwise_probability(risk_a, risk_b)

        self.assertEqual(result.tie_rate, 0.5)
        self.assertEqual(result.a_win_rate, 0.25)
        self.assertEqual(result.b_win_rate, 0.25)
        self.assertLess(abs(result.p_ab + result.p_ba - 1.0), 1e-9)

    def test_missing_actual_or_model_column_raises_clear_error(self):
        no_actual = pd.DataFrame({"LSTM": [1.0], "Transformer": [2.0]})
        no_model = pd.DataFrame({"Actual": [1.0], "LSTM": [1.0]})

        with self.assertRaisesRegex(ValueError, "Actual"):
            rms.validate_prediction_columns(no_actual, ["LSTM"])
        with self.assertRaisesRegex(ValueError, "Transformer"):
            rms.validate_prediction_columns(no_model, ["LSTM", "Transformer"])

    def test_selected_models_detects_current_columns_and_excludes_removed_baselines(self):
        columns = [
            "Date",
            "Actual",
            "LSTM",
            "Transformer",
            "TCN",
            "Stacking",
            "DynamicGatedOnly",
            "AdaptiveWeightedStacking",
            "DynamicGatedStacking",
            "DynamicGatedStacking_PI95_Lower",
            "Aquifer",
        ]

        models = rms.selected_models(available_columns=columns)

        self.assertEqual(models, ["LSTM", "Transformer", "TCN", "DynamicGatedStacking"])
        self.assertNotIn("Persistence", models)
        self.assertNotIn("Stacking", models)
        self.assertNotIn("AdaptiveWeightedStacking", models)
        self.assertNotIn("DynamicGatedOnly", models)

    def test_str2bool_parses_expected_values(self):
        for value in ["true", "1", "yes", "y", "t"]:
            self.assertTrue(rms.str2bool(value))
        for value in ["false", "0", "no", "n", "f"]:
            self.assertFalse(rms.str2bool(value))
        with self.assertRaises(argparse.ArgumentTypeError):
            rms.str2bool("maybe")

    def test_rejections_allow_one_rejected_model_to_have_multiple_dominators(self):
        pairwise = pd.DataFrame(
            [
                {
                    "split": "test",
                    "well": "well1",
                    "aquifer": "A",
                    "model_a": "LSTM",
                    "model_b": "TCN",
                    "p_a_better_than_b": 0.96,
                    "dominates": True,
                    "loss": "squared",
                },
                {
                    "split": "test",
                    "well": "well1",
                    "aquifer": "A",
                    "model_a": "Transformer",
                    "model_b": "TCN",
                    "p_a_better_than_b": 0.982,
                    "dominates": True,
                    "loss": "squared",
                },
            ]
        )

        rejections = rms.build_stable_rejections(pairwise, 0.95)

        self.assertEqual(len(rejections), 2)
        self.assertEqual(set(rejections["dominating_model"]), {"LSTM", "Transformer"})
        self.assertEqual(set(rejections["rejected_model"]), {"TCN"})
        self.assertIn("P(R_Transformer < R_TCN) = 0.982 >= 0.95", set(rejections["reason"]))

    def test_ranking_distinguishes_pair_and_scenario_level_counts(self):
        risk_summary = pd.DataFrame(
            [
                {"split": "test", "well": "well1", "model": "LSTM", "mean_risk": 1.0},
                {"split": "test", "well": "well1", "model": "Transformer", "mean_risk": 1.1},
                {"split": "test", "well": "well1", "model": "TCN", "mean_risk": 3.0},
            ]
        )
        pairwise = pd.DataFrame(
            [
                {
                    "split": "test",
                    "well": "well1",
                    "aquifer": "A",
                    "model_a": "LSTM",
                    "model_b": "TCN",
                    "p_a_better_than_b": 0.96,
                    "dominates": True,
                    "loss": "squared",
                },
                {
                    "split": "test",
                    "well": "well1",
                    "aquifer": "A",
                    "model_a": "Transformer",
                    "model_b": "TCN",
                    "p_a_better_than_b": 0.97,
                    "dominates": True,
                    "loss": "squared",
                },
                {
                    "split": "test",
                    "well": "well1",
                    "aquifer": "A",
                    "model_a": "TCN",
                    "model_b": "LSTM",
                    "p_a_better_than_b": 0.04,
                    "dominates": False,
                    "loss": "squared",
                },
            ]
        )

        ranking = rms.build_model_ranking(
            risk_summary,
            pairwise,
            models=["LSTM", "Transformer", "TCN"],
            loss="squared",
            bootstrap_method="block",
            block_size=8,
        )
        tcn = ranking.loc[(ranking["split"] == "test") & (ranking["model"] == "TCN")].iloc[0]

        self.assertEqual(tcn["stable_loss_count"], 2)
        self.assertEqual(tcn["rejected_count"], 1)
        self.assertEqual(tcn["dominance_count"], 2)
        self.assertIn("stable_win_count", ranking.columns)
        self.assertIn("stable_loss_count", ranking.columns)

    def test_run_analysis_creates_csv_outputs_with_current_model_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            well_dir = out_dir / "well1"
            well_dir.mkdir()
            frame = pd.DataFrame(
                {
                    "Date": pd.date_range("2020-01-01", periods=12, freq="W"),
                    "Actual": np.arange(12, dtype=float),
                    "LSTM": np.arange(12, dtype=float),
                    "Transformer": np.arange(12, dtype=float) + 0.1,
                    "TCN": np.arange(12, dtype=float) + 2.0,
                    "Stacking": np.arange(12, dtype=float) + 0.2,
                    "DynamicGatedStacking": np.arange(12, dtype=float) + 0.03,
                    "Aquifer": ["A"] * 12,
                }
            )
            frame.to_csv(well_dir / "test_predictions.csv", index=False)

            rms.run_analysis(
                out_dir=out_dir,
                splits=["test"],
                loss="squared",
                bootstrap_samples=50,
                bootstrap_method="iid",
                block_size=4,
                dominance_threshold=0.95,
                trend_threshold=0.70,
                exclude_models=["Persistence", "Stacking", "DynamicGatedOnly", "AdaptiveWeightedStacking"],
                target_model="DynamicGatedStacking",
                save_risk_samples=True,
                quantile_grid_size=11,
                seed=42,
            )

            result_dir = out_dir / "reproducible_selection"
            ranking = pd.read_csv(result_dir / "model_reproducibility_ranking.csv")
            pairwise = pd.read_csv(result_dir / "pairwise_reproducible_dominance.csv")
            rejections = pd.read_csv(result_dir / "stable_model_rejections.csv")
            quantiles = pd.read_csv(result_dir / "loss_quantile_functions.csv")

            self.assertFalse((ranking["model"] == "Persistence").any())
            self.assertFalse((pairwise["model_a"] == "Persistence").any())
            self.assertFalse((ranking["model"] == "Stacking").any())
            self.assertFalse((pairwise["model_a"] == "Stacking").any())
            self.assertFalse((ranking["model"] == "AdaptiveWeightedStacking").any())
            self.assertFalse((ranking["model"] == "DynamicGatedOnly").any())
            self.assertTrue((result_dir / "risk_distribution_summary.csv").exists())
            self.assertTrue((result_dir / "risk_distribution_samples.csv").exists())
            self.assertTrue((result_dir / "loss_quantile_functions.csv").exists())
            self.assertTrue((result_dir / "pairwise_dominance_probabilities.csv").exists())
            self.assertTrue((result_dir / "pairwise_probability_heatmap_test.png").exists())
            self.assertTrue((result_dir / "risk_distribution_plot_test.png").exists())
            self.assertTrue((result_dir / "dynamic_gated_reproducibility_report.md").exists())
            self.assertIn("dominating_model", rejections.columns)
            dgs_quantiles = quantiles[quantiles["model"] == "DynamicGatedStacking"]["loss_quantile"].to_numpy()
            self.assertTrue(np.all(np.diff(dgs_quantiles) >= -1e-12))

    def test_run_analysis_records_effective_block_size_when_series_is_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            well_dir = out_dir / "well1"
            well_dir.mkdir()
            frame = pd.DataFrame(
                {
                    "Date": pd.date_range("2020-01-01", periods=3, freq="W"),
                    "Actual": np.array([1.0, 2.0, 3.0]),
                    "LSTM": np.array([1.0, 2.0, 3.0]),
                    "Transformer": np.array([1.1, 1.9, 3.2]),
                    "TCN": np.array([2.0, 3.0, 4.0]),
                    "Stacking": np.array([1.0, 2.1, 3.1]),
                    "DynamicGatedStacking": np.array([1.0, 2.0, 3.0]),
                    "Aquifer": ["A"] * 3,
                }
            )
            frame.to_csv(well_dir / "test_predictions.csv", index=False)

            with self.assertWarnsRegex(UserWarning, "block_size"):
                rms.run_analysis(
                    out_dir=out_dir,
                    splits=["test"],
                    loss="squared",
                    bootstrap_samples=10,
                    bootstrap_method="block",
                    block_size=8,
                    dominance_threshold=0.95,
                    trend_threshold=0.70,
                    exclude_models=["Persistence", "Stacking", "DynamicGatedOnly", "AdaptiveWeightedStacking"],
                    target_model="DynamicGatedStacking",
                    save_risk_samples=False,
                    quantile_grid_size=11,
                    seed=42,
                )

            result_dir = out_dir / "reproducible_selection"
            risk_summary = pd.read_csv(result_dir / "risk_distribution_summary.csv")
            ranking = pd.read_csv(result_dir / "model_reproducibility_ranking.csv")

            self.assertEqual(set(risk_summary["block_size"]), {3})
            self.assertEqual(set(ranking["block_size"]), {3})

    def test_default_splits_only_load_test_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            well_dir = out_dir / "well1"
            well_dir.mkdir()
            test_frame = pd.DataFrame(
                {
                    "Actual": np.arange(5, dtype=float),
                    "LSTM": np.arange(5, dtype=float),
                    "Transformer": np.arange(5, dtype=float) + 0.1,
                    "TCN": np.arange(5, dtype=float) + 0.2,
                    "DynamicGatedStacking": np.arange(5, dtype=float) + 0.03,
                }
            )
            future_frame = test_frame.drop(columns=["DynamicGatedStacking"])
            test_frame.to_csv(well_dir / "test_predictions.csv", index=False)
            future_frame.to_csv(well_dir / "future_holdout_predictions.csv", index=False)

            rms.run_analysis(
                out_dir=out_dir,
                splits=["test"],
                loss="squared",
                bootstrap_samples=10,
                bootstrap_method="iid",
                block_size=2,
                dominance_threshold=0.95,
                trend_threshold=0.70,
                exclude_models=["Persistence", "Stacking", "DynamicGatedOnly", "AdaptiveWeightedStacking"],
                target_model="DynamicGatedStacking",
                save_risk_samples=False,
                quantile_grid_size=5,
                seed=42,
            )

            summary = pd.read_csv(out_dir / "reproducible_selection" / "risk_distribution_summary.csv")
            self.assertEqual(set(summary["split"]), {"test"})

    def test_prepare_result_dir_removes_old_split_heatmaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "reproducible_selection"
            per_well_dir = result_dir / "per_well_heatmaps"
            per_well_dir.mkdir(parents=True)
            stale_split = result_dir / "pairwise_probability_heatmap_future_holdout.png"
            stale_custom_split = result_dir / "pairwise_probability_heatmap_custom.png"
            unrelated = result_dir / "notes.txt"
            stale_per_well = per_well_dir / "pairwise_probability_heatmap_custom_well1.png"
            stale_split.write_bytes(b"old")
            stale_custom_split.write_bytes(b"old")
            unrelated.write_text("keep", encoding="utf-8")
            stale_per_well.write_bytes(b"old")

            rms._prepare_result_dir(result_dir)

            self.assertFalse(stale_split.exists())
            self.assertFalse(stale_custom_split.exists())
            self.assertFalse(stale_per_well.exists())
            self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
