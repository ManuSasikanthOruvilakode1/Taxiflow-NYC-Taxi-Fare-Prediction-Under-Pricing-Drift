"""Regression metrics for fare prediction."""

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE, median absolute error and mean bias (prediction - truth), in dollars."""
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    if err.size == 0:
        return {"n": 0, "rmse": None, "mae": None, "medae": None, "bias": None}
    abs_err = np.abs(err)
    return {
        "n": int(err.size),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(abs_err)),
        "medae": float(np.median(abs_err)),
        "bias": float(np.mean(err)),
    }
