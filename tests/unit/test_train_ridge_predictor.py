from __future__ import annotations

import unittest

import numpy as np

from benchmarks.train_ridge_predictor import (
    _feature_group_folds,
    _metrics,
    _oof_predictions,
    _predict,
    fit_ridge,
    select_alpha,
)


class TrainRidgePredictorTests(unittest.TestCase):
    def test_fit_uses_unpenalized_intercept_and_standardized_features(self) -> None:
        x = np.asarray([[1.0, 10.0], [2.0, 8.0], [3.0, 6.0], [4.0, 4.0], [5.0, 1.0]])
        y = 0.25 + 0.03 * x[:, 0] + 0.02 * x[:, 1]
        intercept, coefficients, mean, scale = fit_ridge(x, y, 0.0)
        prediction = _predict(x, intercept, coefficients, mean, scale)
        np.testing.assert_allclose(prediction, y, rtol=1e-10, atol=1e-10)
        self.assertAlmostEqual(intercept, float(y.mean()))

    def test_feature_group_folds_keep_duplicates_together(self) -> None:
        x = np.asarray(
            [[1.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=np.float64
        )
        folds = _feature_group_folds(x, fold_count=3)
        membership = {
            row: fold_index
            for fold_index, fold in enumerate(folds)
            for row in fold.tolist()
        }
        self.assertEqual(membership[0], membership[1])
        self.assertEqual(sorted(membership), list(range(len(x))))

    def test_alpha_selection_and_oof_cover_every_row(self) -> None:
        x = np.arange(24, dtype=np.float64).reshape(12, 2)
        y = 0.1 + x[:, 0] * 0.002 - x[:, 1] * 0.001
        folds = [np.arange(offset, len(y), 3) for offset in range(3)]
        alpha, candidates = select_alpha(x, y, folds, (0.0, 1.0, 100.0))
        self.assertEqual(alpha, 0.0)
        self.assertEqual(len(candidates), 3)
        prediction = _oof_predictions(x, y, folds, alpha)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertLess(_metrics(y, prediction)["mae_seconds"], 1e-10)


if __name__ == "__main__":
    unittest.main()
