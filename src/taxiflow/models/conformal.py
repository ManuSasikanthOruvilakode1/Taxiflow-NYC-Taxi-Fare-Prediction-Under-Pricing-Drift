"""Split Conformal Prediction for distribution-free uncertainty intervals."""

import numpy as np


class ConformalPredictor:
    """Split Conformal Prediction wrapper providing finite-sample coverage guarantees.

    Given a desired miscoverage level alpha in (0, 1) (e.g. alpha=0.10 for 90% confidence),
    the conformal predictor computes non-conformity scores on a held-out calibration set
    and takes their finite-sample-corrected quantile q_hat.

    Scores are |y - y_hat| / scale. With scale=None (constant 1) intervals have a fixed
    width of +/- q_hat. Passing a per-sample difficulty estimate as `scale` (e.g. a model
    of expected absolute error) gives locally adaptive intervals y_hat +/- q_hat * scale;
    the marginal coverage guarantee holds either way, provided calibration and test data
    are exchangeable.
    """

    def __init__(self, confidence_level: float = 0.90):
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.q_hat: float | None = None

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray, scale: np.ndarray | None = None) -> float:
        """Calibrate on held-out data. Returns q_hat."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have identical length")

        n = len(y_true)
        if n == 0:
            raise ValueError("Calibration set cannot be empty")

        scores = np.abs(y_true - y_pred)
        if scale is not None:
            scores = scores / np.asarray(scale, dtype=np.float64)
        # Finite-sample correction: ceil((n + 1) * (1 - alpha)) / n
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)

        self.q_hat = float(np.quantile(scores, level, method="higher"))
        return self.q_hat

    def predict_interval(self, y_pred, min_val: float = 3.00, scale=None):
        """Prediction interval [lower, upper] for point prediction(s).

        Args:
            y_pred: Point prediction(s)
            min_val: Floor for the lower bound (NYC metered base fare is $3.00)
            scale: Per-sample difficulty, required if calibrated with one

        Returns:
            Tuple of (lower_bounds, upper_bounds), scalars if y_pred is a scalar.
        """
        if self.q_hat is None:
            raise RuntimeError("ConformalPredictor must be calibrated before predicting intervals")

        is_scalar = np.isscalar(y_pred)
        y_arr = np.atleast_1d(np.asarray(y_pred, dtype=np.float64))
        half_width = self.q_hat if scale is None else self.q_hat * np.atleast_1d(scale)

        lower = np.maximum(min_val, y_arr - half_width)
        upper = y_arr + half_width

        if is_scalar:
            return float(lower[0]), float(upper[0])
        return lower, upper

    def evaluate_coverage(self, y_test: np.ndarray, y_pred_test: np.ndarray, scale=None) -> tuple[float, float]:
        """Empirical coverage and mean interval width on test data."""
        lower, upper = self.predict_interval(y_pred_test, scale=scale)
        in_bounds = (y_test >= lower) & (y_test <= upper)
        return float(np.mean(in_bounds)), float(np.mean(upper - lower))
