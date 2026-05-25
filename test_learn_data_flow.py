import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import learn
import reproducible_model_selection as rms


FEATURES = ["GWL", "TASMAX", "TAS", "TASMIN", "Humidity", "Precipitation"]


def make_frame(n: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2000-01-03", periods=n, freq="7D")
    data = {"Date": dates}
    for i, name in enumerate(FEATURES):
        data[name] = np.arange(n, dtype=float) + i * 100.0
    return pd.DataFrame(data)


class StrictSplitTests(unittest.TestCase):
    def test_prepare_splits_orders_selection_calib_test_before_future_holdout(self):
        df = make_frame(80)
        split = learn.prepare_splits(
            df,
            features=FEATURES,
            target="GWL",
            lookback=4,
            horizon=1,
            train_ratio=0.4,
            val_ratio=0.1,
            selection_ratio=0.1,
            calib_ratio=0.1,
            holdout_steps=5,
        )

        self.assertEqual(len(split.y_future_holdout), 5)
        self.assertTrue(np.array_equal(split.dates_future_holdout, df["Date"].tail(5).values))
        self.assertLess(split.dates_test[-1], split.dates_future_holdout[0])
        self.assertLess(split.dates_selection[-1], split.dates_calib[0])
        self.assertLess(split.dates_calib[-1], split.dates_test[0])
        self.assertTrue(np.array_equal(split.idx_future_holdout, np.arange(75, 80)))

    def test_scaler_fit_only_sees_training_visible_rows(self):
        df = make_frame(80)
        split = learn.prepare_splits(
            df,
            features=FEATURES,
            target="GWL",
            lookback=4,
            horizon=1,
            train_ratio=0.4,
            val_ratio=0.1,
            selection_ratio=0.1,
            calib_ratio=0.1,
            holdout_steps=5,
        )

        train_sequence_count = len(split.X_train)
        train_end_idx = 4 + train_sequence_count + 1 - 2
        expected_mean = df[FEATURES].to_numpy()[: train_end_idx + 1].mean(axis=0)
        np.testing.assert_allclose(split.scaler.mean_, expected_mean)

    def test_future_holdout_known_features_are_real_holdout_weather(self):
        df = make_frame(80)
        split = learn.prepare_splits(
            df,
            features=FEATURES,
            target="GWL",
            lookback=4,
            horizon=1,
            train_ratio=0.4,
            val_ratio=0.1,
            selection_ratio=0.1,
            calib_ratio=0.1,
            holdout_steps=5,
        )

        scaled_holdout = split.scaler.transform(df[FEATURES].to_numpy()[75:80])
        np.testing.assert_allclose(split.future_known_features_scaled[:, 1:], scaled_holdout[:, 1:])

    def test_target_scaling_uses_target_column_statistics(self):
        df = make_frame(80)
        split = learn.prepare_splits(
            df,
            features=FEATURES,
            target="GWL",
            lookback=4,
            horizon=1,
            train_ratio=0.4,
            val_ratio=0.1,
            selection_ratio=0.1,
            calib_ratio=0.1,
            holdout_steps=5,
        )
        raw = np.array([49.0, 50.0, 51.0])

        scaled = learn.scale_target_values(split.scaler, raw)

        expected = (raw - split.scaler.mean_[0]) / split.scaler.scale_[0]
        np.testing.assert_allclose(scaled, expected)


class TCNConfigurationTests(unittest.TestCase):
    def test_parse_tcn_channels_rejects_empty_or_non_positive_values(self):
        self.assertEqual(learn.parse_tcn_channels("32, 32,64"), [32, 32, 64])

        with self.assertRaisesRegex(ValueError, "positive"):
            learn.parse_tcn_channels("32,0,64")
        with self.assertRaisesRegex(ValueError, "at least one"):
            learn.parse_tcn_channels("")

    def test_default_tcn_receptive_field_covers_default_lookback(self):
        channels = learn.parse_tcn_channels(learn.DEFAULT_TCN_CHANNELS)
        receptive_field = learn.tcn_receptive_field(kernel=learn.DEFAULT_TCN_KERNEL, n_layers=len(channels))

        self.assertEqual(channels, [32, 32, 32, 32])
        self.assertEqual(receptive_field, 31)
        self.assertGreaterEqual(receptive_field, 18)

    def test_stacking_xgb_uses_requested_seed(self):
        xgb = learn.create_stacking_xgb(seed=123)

        self.assertEqual(xgb.get_params()["random_state"], 123)

    def test_dynamic_residual_xgb_uses_requested_seed_and_weak_defaults(self):
        xgb = learn.create_dynamic_residual_xgb(seed=123)
        params = xgb.get_params()

        self.assertEqual(params["random_state"], 123)
        self.assertLessEqual(params["max_depth"], 2)
        self.assertLessEqual(params["n_estimators"], 100)
        self.assertGreaterEqual(params["reg_lambda"], 1.0)


class DynamicGatedStackingTests(unittest.TestCase):
    def test_mc_dropout_predict_averages_stochastic_passes(self):
        class CountingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x):
                self.calls += 1
                return torch.full((x.shape[0],), float(self.calls), dtype=torch.float32, device=x.device)

        model = CountingModel()
        model.eval()
        X = np.zeros((2, 3, 1), dtype=float)

        mean, std = learn.mc_dropout_predict(model, X, torch.device("cpu"), samples=3)

        np.testing.assert_allclose(mean, np.array([2.0, 2.0]))
        np.testing.assert_allclose(std, np.std(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), axis=0))
        self.assertFalse(model.training)

    def test_dynamic_gate_weights_are_probabilities(self):
        gate = learn.DynamicGateRegressor(input_dim=6, hidden=4, n_models=3)
        x = torch.zeros((5, 6), dtype=torch.float32)

        weights = gate(x)

        self.assertEqual(tuple(weights.shape), (5, 3))
        self.assertTrue(torch.all(weights >= 0.0))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(5))

    def test_residual_correction_adds_xgboost_residual_prediction(self):
        class ResidualModel:
            def predict(self, features):
                return np.full(features.shape[0], 0.25)

        gated = np.array([1.0, 2.0, 3.0])
        features = np.zeros((3, 4))

        corrected = learn.apply_residual_correction(gated, ResidualModel(), features)

        np.testing.assert_allclose(corrected, np.array([1.25, 2.25, 3.25]))

    def test_dynamic_residual_selection_rejects_worse_residual(self):
        decision = learn.choose_dynamic_residual_usage(
            y_true=np.array([1.0, 2.0, 3.0]),
            gated_only_prediction=np.array([1.0, 2.0, 3.0]),
            residual_prediction=np.array([2.0, 3.0, 4.0]),
        )

        self.assertFalse(decision["use_residual"])

    def test_dynamic_residual_selection_accepts_better_residual(self):
        decision = learn.choose_dynamic_residual_usage(
            y_true=np.array([1.0, 2.0, 3.0]),
            gated_only_prediction=np.array([2.0, 3.0, 4.0]),
            residual_prediction=np.array([1.0, 2.0, 3.0]),
        )

        self.assertTrue(decision["use_residual"])

    def test_dynamic_gated_stacking_is_in_reproducible_model_lists(self):
        self.assertIn(learn.DYNAMIC_MODEL_NAME, rms.DEFAULT_MODELS)
        self.assertIn(learn.DYNAMIC_MODEL_NAME, rms.selected_models(
            available_columns=["Actual", "LSTM", "Transformer", "TCN", learn.DYNAMIC_MODEL_NAME]
        ))
        self.assertNotIn(learn.DYNAMIC_MODEL_NAME, rms.DEFAULT_EXCLUDE_MODELS)

    def test_dynamic_gate_weight_rows_are_labeled_by_split_and_model(self):
        dates = pd.date_range("2020-01-01", periods=2, freq="W")
        weights = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])

        rows = learn.dynamic_gate_weight_rows("test", dates, weights)

        self.assertEqual(rows[0]["split"], "test")
        self.assertEqual(rows[0]["Date"], dates[0])
        self.assertEqual(rows[0]["weight_LSTM"], 0.2)
        self.assertEqual(rows[0]["weight_Transformer"], 0.3)
        self.assertEqual(rows[0]["weight_TCN"], 0.5)
        self.assertAlmostEqual(rows[1]["weight_sum"], 1.0)


if __name__ == "__main__":
    unittest.main()
