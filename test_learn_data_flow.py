import unittest

import numpy as np
import pandas as pd

import learn
import lookback_experiment
import validate_selected_weekly_data


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

    def test_persistence_baselines_use_split_specific_fairness_rules(self):
        df = make_frame(80)
        test_idx = np.array([50, 51, 52])
        np.testing.assert_allclose(
            learn.persistence_for_indices(df, test_idx, target="GWL"),
            np.array([49.0, 50.0, 51.0]),
        )
        np.testing.assert_allclose(
            learn.recursive_persistence(last_actual=74.0, steps=5),
            np.full(5, 74.0),
        )


class LookbackSelectionTests(unittest.TestCase):
    def test_best_lookback_uses_selection_split_not_test(self):
        summary = pd.DataFrame(
            [
                {"split": "selection", "lookback": 12, "model": "Stacking", "NSE_mean": 0.8, "RMSE_mean": 2.0},
                {"split": "test", "lookback": 12, "model": "Stacking", "NSE_mean": 0.99, "RMSE_mean": 1.0},
                {"split": "selection", "lookback": 18, "model": "Stacking", "NSE_mean": 0.9, "RMSE_mean": 3.0},
                {"split": "test", "lookback": 18, "model": "Stacking", "NSE_mean": 0.1, "RMSE_mean": 10.0},
            ]
        )

        best = lookback_experiment.choose_best_lookback(summary)
        self.assertEqual(best["selection_split"], "selection")
        self.assertEqual(best["best_lookback"], 18)


class SelectedWeeklyValidationTests(unittest.TestCase):
    def test_validate_nine_well_outputs_covers_current_training_dataset(self):
        validate_selected_weekly_data.validate_nine_well_output_files()


if __name__ == "__main__":
    unittest.main()
