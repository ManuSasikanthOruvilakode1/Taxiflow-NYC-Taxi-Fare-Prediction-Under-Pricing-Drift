"""End-to-end training and evaluation orchestration."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from taxiflow.config import AppConfig
from taxiflow.data.download import month_range
from taxiflow.evaluation.metrics import regression_metrics
from taxiflow.features.build import build_features
from taxiflow.models.train import (
    FareModel,
    ZonePairMedian,
    features_for,
    fit_elasticnet,
    fit_fare_model,
    load_window,
    predict_elasticnet,
    seasonal_index_for,
)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
BASELINES_PATH = MODELS_DIR / "baselines.joblib"
BASE_MODEL_PATH = MODELS_DIR / "taxiflow_lgbm.joblib"  # as trained; used for all evaluation
TIMELINE_SAMPLE_PER_MONTH = 100_000
LATENCY_REQUESTS = 2000


def drift_model_path(name: str) -> Path:
    return MODELS_DIR / f"taxiflow_lgbm_{name}.joblib"


def run_train(cfg: AppConfig, log=print) -> None:
    seasonal = seasonal_index_for(cfg)
    log("seasonal index (Jan..Dec): " + " ".join(f"{v:.3f}" for v in seasonal[1:]))

    log("== Main model (current pricing regime) ==")
    s = cfg.split
    model = fit_fare_model(cfg, s.train, s.validation, s.sample_per_month, s.calibration, seasonal, log=log)
    model.save(BASE_MODEL_PATH)
    model.save(cfg.serving.model_path)  # serving copy; `taxiflow correct` adds the monthly correction

    for d in cfg.drift_models:
        log(f"== Drift model: {d.name} ==")
        fit_fare_model(cfg, d.train, d.validation, d.sample_per_month, None, seasonal, log=log).save(
            drift_model_path(d.name))

    log("== Baselines ==")
    base_df = load_window(cfg.data.processed_dir, s.train, s.sample_per_month // 4, cfg.seed)
    pair = ZonePairMedian().fit(base_df)
    enet = fit_elasticnet(features_for(base_df, model.routes), base_df["target"].to_numpy(), cfg.seed)
    joblib.dump({"zone_pair_median": pair, "elasticnet": enet}, BASELINES_PATH, compress=3)
    log(f"saved models to {MODELS_DIR}/")


def _segments(df: pd.DataFrame) -> dict[str, np.ndarray]:
    from taxiflow.features.geospatial import load_zone_table

    zones = load_zone_table()
    pu, do = df["pu_zone"].to_numpy(), df["do_zone"].to_numpy()
    airport = zones.is_airport[pu] | zones.is_airport[do]
    cbd = df["cbd_fee"].to_numpy() > 0
    flex = df["is_flex"].to_numpy()
    return {
        "all": np.ones(len(df), dtype=bool),
        "congestion_fee_charged": cbd,
        "no_congestion_fee": ~cbd,
        "flex_fare": flex,
        "metered": ~flex,
        "airport": airport,
        "non_airport": ~airport,
    }


def _measure_latency(model: FareModel, n: int = LATENCY_REQUESTS, seed: int = 0) -> dict:
    """Single-request inference latency: feature build + point prediction + adaptive interval."""
    rng = np.random.default_rng(seed)
    pickups = np.datetime64("2026-07-01T00:00") + rng.integers(0, 60 * 24 * 60, n).astype("timedelta64[m]")
    pu, do = rng.integers(1, 264, n), rng.integers(1, 264, n)
    timings = []
    for i in range(n):
        t0 = time.perf_counter()
        X = build_features([pu[i]], [do[i]], pickups[i : i + 1], [False], model.routes)
        pred = model.predict(X)
        model.conformal_adaptive.predict_interval(pred, scale=model.scale(X))
        timings.append((time.perf_counter() - t0) * 1000)
    t = np.array(timings[50:])  # drop warm-up
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "p99_ms": float(np.percentile(t, 99)), "n_requests": len(t)}


def run_evaluate(cfg: AppConfig, log=print) -> dict:
    model = FareModel.load(BASE_MODEL_PATH)
    drift = {d.name: FareModel.load(drift_model_path(d.name)) for d in cfg.drift_models}
    baselines = joblib.load(BASELINES_PATH)
    lgbm_models = {"lightgbm": model, **{f"lightgbm_{n}": m for n, m in drift.items()}}

    # ---- Out-of-time test set: every cleaned trip in the test months.
    test_df = load_window(cfg.data.processed_dir, cfg.split.test)
    y = test_df["target"].to_numpy()
    X = features_for(test_df, model.routes)
    log(f"test rows {len(y):,} ({cfg.split.test[0]}..{cfg.split.test[1]})")

    preds = {"lightgbm": model.predict(X)}
    for name, m in lgbm_models.items():
        if m is not model:
            preds[name] = m.predict(features_for(test_df, m.routes))
    preds["elasticnet"] = predict_elasticnet(baselines["elasticnet"], X)
    preds["zone_pair_median"] = baselines["zone_pair_median"].predict(test_df)
    scale = model.scale(X)
    segments = _segments(test_df)

    results = {"test_window": list(cfg.split.test), "n_test": len(y), "models": {}, "segments": {},
               "intervals": {}}
    for name, p in preds.items():
        results["models"][name] = regression_metrics(y, p)
    for seg, mask in segments.items():
        results["segments"][seg] = {"share": float(mask.mean())} | {
            name: regression_metrics(y[mask], preds[name][mask]) for name in lgbm_models
        }
    for kind, conf, sc in (("fixed", model.conformal_fixed, None), ("adaptive", model.conformal_adaptive, scale)):
        per_seg = {}
        for seg, mask in segments.items():
            cov, width = conf.evaluate_coverage(y[mask], preds["lightgbm"][mask], None if sc is None else sc[mask])
            per_seg[seg] = {"coverage": cov, "mean_width": width}
        results["intervals"][kind] = {"q_hat": conf.q_hat, "segments": per_seg}
    results["intervals"]["confidence_level"] = model.conformal_fixed.confidence_level
    del X, test_df

    # ---- Month-by-month timeline for the drift study.
    timeline = []
    for month in month_range(cfg.data.start_month, cfg.data.end_month):
        df = load_window(cfg.data.processed_dir, (month, month), TIMELINE_SAMPLE_PER_MONTH, cfg.seed)
        yt = df["target"].to_numpy()
        row = {"month": month}
        for name, m in lgbm_models.items():
            p = m.predict(features_for(df, m.routes))
            row[name] = {"mae": float(np.mean(np.abs(yt - p))), "bias": float(np.mean(p - yt))}
        timeline.append(row)
    results["timeline"] = timeline

    results["latency"] = _measure_latency(model)
    results["model_metadata"] = model.metadata
    results["seasonal_index"] = {f"{m:02d}": float(model.seasonal_index[m]) for m in range(1, 13)}
    cleaning = json.loads((Path(cfg.data.processed_dir) / "cleaning_report.json").read_text())
    results["data"] = {
        "months": [cfg.data.start_month, cfg.data.end_month],
        "raw_rows": sum(r["raw_rows"] for r in cleaning),
        "clean_rows": sum(r["clean_rows"] for r in cleaning),
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    log(f"wrote {REPORTS_DIR / 'metrics.json'}")
    return results
