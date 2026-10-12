"""Unit tests for Split Conformal Prediction uncertainty guarantees."""

import numpy as np

from taxiflow.models.conformal import ConformalPredictor


def test_conformal_calibration_and_coverage():
    np.random.seed(42)
    n_cal = 1000
    n_test = 500

    # Synthetic true fares and noisy predictions (standard deviation = $3.00)
    y_cal_true = np.random.uniform(10.0, 80.0, size=n_cal)
    y_cal_pred = y_cal_true + np.random.normal(0, 3.0, size=n_cal)

    y_test_true = np.random.uniform(10.0, 80.0, size=n_test)
    y_test_pred = y_test_true + np.random.normal(0, 3.0, size=n_test)

    # Instantiate 90% confidence predictor
    predictor = ConformalPredictor(confidence_level=0.90)
    q_hat = predictor.calibrate(y_cal_true, y_cal_pred)

    assert q_hat > 0.0
    # For normal(0, 3), 90% two-sided quantile is approx 1.645 * 3 ≈ 4.93
    assert 4.0 < q_hat < 6.0

    coverage, avg_width = predictor.evaluate_coverage(y_test_true, y_test_pred)
    # Empirical coverage on test should be close to nominal 90%
    assert 0.87 <= coverage <= 0.94, f"Expected coverage close to 0.90, got {coverage}"
    assert avg_width > 0.0


def test_conformal_scalar_interval():
    predictor = ConformalPredictor(confidence_level=0.90)
    predictor.q_hat = 5.0

    lower, upper = predictor.predict_interval(30.0, min_val=2.50)
    assert lower == 25.0
    assert upper == 35.0

    # Test floor bound with small prediction
    lower_floor, upper_floor = predictor.predict_interval(4.0, min_val=2.50)
    assert lower_floor == 2.50
    assert upper_floor == 9.0
