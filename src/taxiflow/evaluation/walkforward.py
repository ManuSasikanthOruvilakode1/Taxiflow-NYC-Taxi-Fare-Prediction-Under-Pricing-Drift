"""Walk-forward evaluation of model-maintenance strategies.

For each test month T, three strategies are compared on every trip in T:

  static       the deployed model, unchanged.
  bias_fix     the deployed model minus its mean residual in month T-1, estimated separately for
               Flex Fare and metered trips; conformal layers recalibrated on T-1.
  retrain      a fresh model trained on months [T-1-W, T-2] with the deployed model's number of
               boosting rounds; month T-1 is split in half, one half for the interval-width model
               (and nothing else), the other for conformal calibration.

Assumes month T-1 is available at the start of month T (an operator's own trip records); the
public TLC release lags by about two months.
"""

import copy

import numpy as np
import pandas as pd

from taxiflow.config import AppConfig
from taxiflow.data.download import month_add
from taxiflow.evaluation.metrics import regression_metrics
from taxiflow.features.build import RouteStats
from taxiflow.models.train import FareModel, features_for, load_window, new_model, train_booster


def _split_halves(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.random.default_rng(seed).random(n) < 0.5
    return mask, ~mask


def retrain_for_month(cfg: AppConfig, deployed: FareModel, test_month: str, log=print) -> FareModel:
    wf = cfg.walk_forward
    train = (month_add(test_month, -1 - wf.train_months), month_add(test_month, -2))
    last = month_add(test_month, -1)
    d = cfg.data
    train_df = load_window(d.processed_dir, train, wf.sample_per_month, cfg.seed)
    routes = RouteStats.fit(train_df["pu_zone"].to_numpy(), train_df["do_zone"].to_numpy(),
                            train_df["distance_km"].to_numpy(), train_df["duration_min"].to_numpy())
    X_tr, y_tr = features_for(train_df, routes), train_df["target"].to_numpy()
    del train_df
    rounds = deployed.metadata["best_iteration"]
    booster = train_booster(cfg, X_tr, y_tr, deployed.seasonal_index, rounds)
    del X_tr, y_tr

    last_df = load_window(d.processed_dir, (last, last))
    X_last, y_last = features_for(last_df, routes), last_df["target"].to_numpy()
    width_half, cal_half = _split_halves(len(y_last), cfg.seed)
    model = new_model(cfg, booster, routes, deployed.seasonal_index, X_last[width_half], y_last[width_half],
                      {"train_window": list(train), "calibration_window": [last, last], "best_iteration": rounds})
    model.calibrate(X_last[cal_half], y_last[cal_half])
    log(f"  retrained for {test_month}: train {train[0]}..{train[1]}, width/calibration {last}")
    return model


def _evaluate(y, pred, lower, upper, is_flex) -> dict:
    out = {}
    for seg, mask in (("all", np.ones_like(is_flex)), ("flex_fare", is_flex), ("metered", ~is_flex)):
        r = regression_metrics(y[mask], pred[mask])
        r["coverage"] = float(np.mean((y[mask] >= lower[mask]) & (y[mask] <= upper[mask])))
        r["mean_width"] = float(np.mean(upper[mask] - lower[mask]))
        out[seg] = r
    return out


def run_walkforward(cfg: AppConfig, deployed: FareModel, log=print) -> dict:
    results = {"assumption": "month T-1 available at start of month T", "months": {}}
    pooled: dict[str, list] = {}
    for test_month in cfg.walk_forward.test_months:
        log(f"walk-forward test month {test_month}")
        last = month_add(test_month, -1)
        test_df = load_window(cfg.data.processed_dir, (test_month, test_month))
        last_df = load_window(cfg.data.processed_dir, (last, last))
        y, is_flex = test_df["target"].to_numpy(), test_df["is_flex"].to_numpy()

        # static
        X = features_for(test_df, deployed.routes)
        pred, scale = deployed.predict(X), deployed.scale(X)
        lo, hi = deployed.conformal_adaptive.predict_interval(pred, scale=scale)
        strategies = {"static": (pred, lo, hi)}

        # bias_fix: per-booking-type offsets from one half of T-1, recalibration on the other half
        fixed = copy.deepcopy(deployed)
        bias = fixed.apply_monthly_correction(features_for(last_df, deployed.routes),
                                              last_df["target"].to_numpy(), last, cfg.seed)
        pred_b = fixed.predict(X)
        lo_b, hi_b = fixed.conformal_adaptive.predict_interval(pred_b, scale=scale)
        strategies["bias_fix"] = (pred_b, lo_b, hi_b)
        del X

        # retrain
        model = retrain_for_month(cfg, deployed, test_month, log)
        X_r = features_for(test_df, model.routes)
        pred_r = model.predict(X_r)
        lo_r, hi_r = model.conformal_adaptive.predict_interval(pred_r, scale=model.scale(X_r))
        strategies["retrain"] = (pred_r, lo_r, hi_r)
        del X_r

        results["months"][test_month] = {
            "bias_estimates": bias,
            "retrain_window": model.metadata["train_window"],
            **{name: _evaluate(y, *s, is_flex) for name, s in strategies.items()},
        }
        for name, (p, lo_, hi_) in strategies.items():
            pooled.setdefault(name, []).append(pd.DataFrame({"y": y, "p": p, "lo": lo_, "hi": hi_, "flex": is_flex}))

    results["pooled"] = {}
    for name, frames in pooled.items():
        df = pd.concat(frames, ignore_index=True)
        results["pooled"][name] = _evaluate(df["y"].to_numpy(), df["p"].to_numpy(), df["lo"].to_numpy(),
                                            df["hi"].to_numpy(), df["flex"].to_numpy())
    return results
