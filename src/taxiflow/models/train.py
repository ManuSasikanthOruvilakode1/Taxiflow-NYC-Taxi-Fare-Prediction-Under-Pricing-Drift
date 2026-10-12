"""Model fitting: LightGBM fare model, conformal calibration and baselines."""

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from taxiflow.config import AppConfig, MonthRange
from taxiflow.data.download import month_range
from taxiflow.features.build import (
    CATEGORICAL,
    FEATURES,
    FLEX_COL,
    MODEL_COLS,
    MODEL_FEATURES,
    MONTH_COL,
    RouteStats,
    build_features,
)
from taxiflow.features.seasonality import fit_seasonal_index
from taxiflow.models.conformal import ConformalPredictor

MIN_SCALE = 0.5  # floor for the interval-width model, in dollars


def load_window(processed_dir: str | Path, window: MonthRange, sample_per_month: int | None = None,
                seed: int = 42) -> pd.DataFrame:
    """Load cleaned trips for an inclusive month window, optionally sampling rows per month."""
    frames = []
    for month in month_range(*window):
        path = (Path(processed_dir) / f"clean_{month}.parquet").as_posix()
        sample = f"USING SAMPLE reservoir({sample_per_month} ROWS) REPEATABLE ({seed})" if sample_per_month else ""
        frames.append(duckdb.sql(f"SELECT * FROM read_parquet('{path}') {sample}").df())
    return pd.concat(frames, ignore_index=True)


def features_for(df: pd.DataFrame, routes: RouteStats) -> np.ndarray:
    return build_features(df["pu_zone"], df["do_zone"], df["pickup_datetime"].to_numpy(), df["is_flex"], routes)


def _threads(X: np.ndarray) -> int:
    return 1 if len(X) < 1000 else 0


@dataclass
class FareModel:
    """Everything needed to serve a prediction: booster, route lookups and calibrators.

    All methods take the full feature matrix from `build_features` (FEATURES order).
    Prediction = booster(model features) * seasonal_index[month] - bias_offsets[is_flex].
    """

    booster: lgb.Booster
    routes: RouteStats
    width_booster: lgb.Booster
    conformal_fixed: ConformalPredictor
    conformal_adaptive: ConformalPredictor
    metadata: dict = field(default_factory=dict)
    seasonal_index: np.ndarray = field(default_factory=lambda: np.ones(13))
    bias_offsets: tuple[float, float] = (0.0, 0.0)  # (metered, flex fare), in dollars

    def seasonal(self, X: np.ndarray) -> np.ndarray:
        return self.seasonal_index[X[:, MONTH_COL].astype(np.int64)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        base = self.booster.predict(X[:, MODEL_COLS], num_threads=_threads(X)) * self.seasonal(X)
        offsets = np.where(X[:, FLEX_COL] > 0, self.bias_offsets[1], self.bias_offsets[0])
        return base - offsets

    def scale(self, X: np.ndarray) -> np.ndarray:
        s = self.width_booster.predict(X[:, MODEL_COLS], num_threads=_threads(X))
        return np.maximum(s, MIN_SCALE)

    def calibrate(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Calibrate both conformal layers on held-out data. Returns (fixed q_hat, adaptive q_hat)."""
        pred = self.predict(X)
        q_fixed = self.conformal_fixed.calibrate(y, pred)
        q_adapt = self.conformal_adaptive.calibrate(y, pred, scale=self.scale(X))
        self.metadata["n_calibration"] = len(y)
        return q_fixed, q_adapt

    def apply_monthly_correction(self, X: np.ndarray, y: np.ndarray, month: str, seed: int = 42) -> dict:
        """Set per-booking-type bias offsets from a recent month and recalibrate the intervals.

        The month is split at random: one half estimates the offsets, the other recalibrates
        the conformal layers on the corrected predictions, so the coverage guarantee holds.
        """
        est = np.random.default_rng(seed).random(len(y)) < 0.5
        self.bias_offsets = (0.0, 0.0)
        resid = self.predict(X[est]) - y[est]
        flex = X[est, FLEX_COL] > 0
        self.bias_offsets = tuple(float(resid[flex == f].mean()) if (flex == f).any() else 0.0
                                  for f in (False, True))
        self.calibrate(X[~est], y[~est])
        self.metadata["correction_month"] = month
        return {"metered": self.bias_offsets[0], "flex_fare": self.bias_offsets[1]}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str | Path) -> "FareModel":
        return joblib.load(path)


def lgb_dataset(X: np.ndarray, y: np.ndarray, reference=None) -> lgb.Dataset:
    """LightGBM dataset over the model features of a full feature matrix."""
    return lgb.Dataset(X[:, MODEL_COLS], y, feature_name=MODEL_FEATURES, categorical_feature=CATEGORICAL,
                       reference=reference, free_raw_data=False)


def seasonal_index_for(cfg: AppConfig) -> np.ndarray:
    if not cfg.seasonality.enabled:
        return np.ones(13)
    return fit_seasonal_index(cfg.data.processed_dir, cfg.seasonality.fit_window)


def train_booster(cfg: AppConfig, X: np.ndarray, y: np.ndarray, seasonal_index: np.ndarray,
                  num_boost_round: int, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                  ) -> lgb.Booster:
    """Fit on the seasonally adjusted target y / index[month]; early-stop if validation data is given."""
    month_factor = seasonal_index[X[:, MONTH_COL].astype(np.int64)]
    dtrain = lgb_dataset(X, y / month_factor)
    if X_val is None:
        return lgb.train(cfg.model.params, dtrain, num_boost_round=num_boost_round)
    val_factor = seasonal_index[X_val[:, MONTH_COL].astype(np.int64)]
    dval = lgb_dataset(X_val, y_val / val_factor, reference=dtrain)
    return lgb.train(
        cfg.model.params, dtrain, num_boost_round=num_boost_round, valid_sets=[dval],
        callbacks=[lgb.early_stopping(cfg.model.early_stopping_rounds, verbose=False), lgb.log_evaluation(200)],
    )


def new_model(cfg: AppConfig, booster: lgb.Booster, routes: RouteStats, seasonal_index: np.ndarray,
              X_width: np.ndarray, y_width: np.ndarray, metadata: dict) -> FareModel:
    """Assemble a FareModel and fit its interval-width model on held-out (X_width, y_width)."""
    model = FareModel(
        booster=booster,
        routes=routes,
        width_booster=None,
        conformal_fixed=ConformalPredictor(cfg.conformal.confidence_level),
        conformal_adaptive=ConformalPredictor(cfg.conformal.confidence_level),
        metadata=metadata,
        seasonal_index=seasonal_index,
    )
    # Interval-width model: expected |residual| from the same features, on data the booster never saw.
    abs_resid = np.abs(y_width - model.predict(X_width))
    width_params = {**cfg.model.params, "num_leaves": 63, "learning_rate": 0.1}
    model.width_booster = lgb.train(width_params, lgb_dataset(X_width, abs_resid), num_boost_round=200)
    return model


def fit_fare_model(cfg: AppConfig, train: MonthRange, validation: MonthRange, sample_per_month: int,
                   calibration: MonthRange | None = None, seasonal_index: np.ndarray | None = None,
                   log=print) -> FareModel:
    """Train the booster on `train`, early-stop on `validation`, fit the interval-width model
    on `validation` residuals and calibrate conformal intervals on `calibration`."""
    d = cfg.data
    if seasonal_index is None:
        seasonal_index = seasonal_index_for(cfg)
    train_df = load_window(d.processed_dir, train, sample_per_month, cfg.seed)
    val_df = load_window(d.processed_dir, validation, sample_per_month, cfg.seed)
    log(f"train rows {len(train_df):,} ({train[0]}..{train[1]}), validation rows {len(val_df):,}")

    routes = RouteStats.fit(train_df["pu_zone"].to_numpy(), train_df["do_zone"].to_numpy(),
                            train_df["distance_km"].to_numpy(), train_df["duration_min"].to_numpy())
    X_tr, y_tr = features_for(train_df, routes), train_df["target"].to_numpy()
    X_va, y_va = features_for(val_df, routes), val_df["target"].to_numpy()
    del train_df

    booster = train_booster(cfg, X_tr, y_tr, seasonal_index, cfg.model.num_boost_round, X_va, y_va)
    log(f"best iteration {booster.best_iteration}, "
        f"validation RMSE (seasonally adjusted) {booster.best_score['valid_0']['rmse']:.3f}")

    model = new_model(cfg, booster, routes, seasonal_index, X_va, y_va, {
        "train_window": list(train),
        "validation_window": list(validation),
        "calibration_window": list(calibration) if calibration else None,
        "seasonality_fit_window": list(cfg.seasonality.fit_window) if cfg.seasonality.enabled else None,
        "n_train": len(y_tr),
        "best_iteration": int(booster.best_iteration),
        "features": MODEL_FEATURES,
    })
    if calibration:
        cal_df = load_window(d.processed_dir, calibration, None)
        q_fixed, q_adapt = model.calibrate(features_for(cal_df, routes), cal_df["target"].to_numpy())
        log(f"conformal q_hat: fixed ${q_fixed:.2f}, adaptive {q_adapt:.3f} x predicted |error|")
    return model


# ---------------------------------------------------------------- baselines


class ZonePairMedian:
    """Historical median fare per (pickup, dropoff) pair, falling back to pickup-zone then global median."""

    def fit(self, df: pd.DataFrame) -> "ZonePairMedian":
        self.pair = df.groupby(["pu_zone", "do_zone"])["target"].median()
        self.pu = df.groupby("pu_zone")["target"].median()
        self.global_ = float(df["target"].median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([df["pu_zone"], df["do_zone"]])
        pred = self.pair.reindex(idx).to_numpy()
        fallback = self.pu.reindex(df["pu_zone"]).to_numpy()
        pred = np.where(np.isnan(pred), fallback, pred)
        return np.where(np.isnan(pred), self.global_, pred)


LINEAR_NUMERIC = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_rush_hour", "is_overnight",
                  "haversine_km", "manhattan_km", "pu_airport", "do_airport", "pu_cbd", "do_cbd", "same_zone",
                  "is_flex", "route_median_km", "route_median_min"]
LINEAR_CATEGORICAL = ["pu_borough", "do_borough"]


def _linear_frame(X: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(X, columns=FEATURES)[LINEAR_NUMERIC + LINEAR_CATEGORICAL]
    # Unseen routes: fall back to straight-line geometry.
    frame["route_median_km"] = frame["route_median_km"].fillna(frame["manhattan_km"])
    frame["route_median_min"] = frame["route_median_min"].fillna(frame["manhattan_km"] * 3.0)
    return frame


def fit_elasticnet(X: np.ndarray, y: np.ndarray, seed: int = 42):
    pre = ColumnTransformer([
        ("num", StandardScaler(), LINEAR_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), LINEAR_CATEGORICAL),
    ])
    model = make_pipeline(pre, ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=seed, max_iter=5000))
    return model.fit(_linear_frame(X), y)


def predict_elasticnet(model, X: np.ndarray) -> np.ndarray:
    return model.predict(_linear_frame(X))
