"""FastAPI production serving layer for Taxiflow fare predictions."""

import math
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from taxiflow import __version__
from taxiflow.features.build import FEATURES, build_features
from taxiflow.features.geospatial import N_ZONES, load_zone_table
from taxiflow.models.train import FareModel

NYC_TZ = ZoneInfo("America/New_York")
HAVERSINE_COL = FEATURES.index("haversine_km")
DEFAULT_MODEL_PATH = "models/taxiflow_lgbm_serving.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.environ.get("TAXIFLOW_MODEL_PATH", DEFAULT_MODEL_PATH)
    if getattr(app.state, "model", None) is None and os.path.exists(path):
        app.state.model = FareModel.load(path)
    load_zone_table()
    yield


app = FastAPI(
    title="Taxiflow NYC Taxi Fare Prediction API",
    description="Upfront NYC yellow taxi fare estimates with split-conformal prediction intervals.",
    version=__version__,
    lifespan=lifespan,
)
app.state.model = None


class FarePredictionRequest(BaseModel):
    pulocation_id: int = Field(..., ge=1, le=N_ZONES, description="Pickup TLC Taxi Zone ID (1-263)")
    dolocation_id: int = Field(..., ge=1, le=N_ZONES, description="Dropoff TLC Taxi Zone ID (1-263)")
    pickup_datetime: datetime | None = Field(
        default=None,
        description="Scheduled pickup time (ISO 8601). Naive times are NYC local; defaults to now.",
    )
    booking_type: Literal["street_hail", "flex_fare"] = Field(
        default="street_hail", description="Metered street hail or app-booked TLC Flex Fare"
    )


class FarePredictionResponse(BaseModel):
    predicted_fare: float = Field(..., description="Expected charge excluding tip, USD")
    confidence_level: float = Field(..., description="Conformal coverage level of the interval")
    interval_lower: float = Field(..., description="Lower bound of the prediction interval, USD")
    interval_upper: float = Field(..., description="Upper bound of the prediction interval, USD")
    route_median_km: float | None = Field(None, description="Historical median trip distance for this zone pair")
    route_median_min: float | None = Field(None, description="Historical median trip duration for this zone pair")
    haversine_distance_km: float = Field(..., description="Straight-line distance between zone centroids")
    is_airport_trip: bool = Field(..., description="Whether trip involves JFK, LGA, or EWR")
    touches_congestion_zone: bool = Field(..., description="Pickup or dropoff in Manhattan below 60th St")
    inference_latency_ms: float = Field(..., description="Server-side feature + model latency in ms")


def _nyc_local(dt: datetime | None) -> np.datetime64:
    if dt is None:
        dt = datetime.now(NYC_TZ)
    if dt.tzinfo is not None:
        dt = dt.astimezone(NYC_TZ).replace(tzinfo=None)
    return np.datetime64(dt, "s")


def _optional(x: float) -> float | None:
    return None if math.isnan(x) else round(float(x), 2)


@app.get("/health")
def healthcheck(request: Request):
    """Liveness and readiness healthcheck."""
    model: FareModel | None = request.app.state.model
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "service": "taxiflow",
        "version": __version__,
        "model_train_window": model.metadata.get("train_window") if model else None,
        "correction_month": model.metadata.get("correction_month") if model else None,
    }


@app.post("/v1/predict-fare", response_model=FarePredictionResponse)
def predict_fare(body: FarePredictionRequest, request: Request):
    """Predict the upfront fare for a trip and return a conformal prediction interval."""
    model: FareModel | None = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded; run `taxiflow train` first")

    start = time.perf_counter()
    pu, do = body.pulocation_id, body.dolocation_id
    X = build_features([pu], [do], np.array([_nyc_local(body.pickup_datetime)]),
                       [body.booking_type == "flex_fare"], model.routes)
    pred = model.predict(X)
    lower, upper = model.conformal_adaptive.predict_interval(pred, scale=model.scale(X))
    latency_ms = (time.perf_counter() - start) * 1000.0

    zones = load_zone_table()
    return FarePredictionResponse(
        predicted_fare=round(float(pred[0]), 2),
        confidence_level=model.conformal_adaptive.confidence_level,
        interval_lower=round(float(lower[0]), 2),
        interval_upper=round(float(upper[0]), 2),
        route_median_km=_optional(model.routes.median_km[pu, do]),
        route_median_min=_optional(model.routes.median_min[pu, do]),
        haversine_distance_km=round(float(X[0, HAVERSINE_COL]), 2),
        is_airport_trip=bool(zones.is_airport[pu] or zones.is_airport[do]),
        touches_congestion_zone=bool(zones.in_cbd[pu] or zones.in_cbd[do]),
        inference_latency_ms=round(latency_ms, 3),
    )
