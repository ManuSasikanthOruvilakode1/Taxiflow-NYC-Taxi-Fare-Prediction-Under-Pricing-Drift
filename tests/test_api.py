"""Integration tests for Taxiflow FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from taxiflow.serving.api import app


@pytest.fixture
def client(fare_model):
    app.state.model = fare_model
    yield TestClient(app)
    app.state.model = None


def test_healthcheck(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "taxiflow"


def test_predict_fare_endpoint(client):
    payload = {
        "pulocation_id": 132,  # JFK
        "dolocation_id": 230,  # Times Square
        "pickup_datetime": "2026-10-15T18:30:00",
    }
    response = client.post("/v1/predict-fare", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["interval_lower"] <= data["predicted_fare"] <= data["interval_upper"]
    assert data["confidence_level"] == pytest.approx(0.9)
    assert data["is_airport_trip"] is True
    assert data["touches_congestion_zone"] is True  # Times Sq is below 60th St
    assert 18.0 < data["haversine_distance_km"] < 25.0
    assert data["inference_latency_ms"] > 0.0


def test_booking_type_changes_prediction(client):
    base = {"pulocation_id": 161, "dolocation_id": 75, "pickup_datetime": "2026-07-01T10:00:00"}
    hail = client.post("/v1/predict-fare", json=base).json()["predicted_fare"]
    flex = client.post("/v1/predict-fare", json={**base, "booking_type": "flex_fare"}).json()["predicted_fare"]
    assert flex != hail


def test_timezone_aware_pickup_is_converted_to_nyc(client):
    utc = {"pulocation_id": 161, "dolocation_id": 75, "pickup_datetime": "2026-07-01T22:00:00Z"}
    local = {"pulocation_id": 161, "dolocation_id": 75, "pickup_datetime": "2026-07-01T18:00:00"}
    assert client.post("/v1/predict-fare", json=utc).json() | {"inference_latency_ms": 0} == \
        client.post("/v1/predict-fare", json=local).json() | {"inference_latency_ms": 0}


@pytest.mark.parametrize("zone", [0, 264, 265, 999])
def test_invalid_zone_id_validation(client, zone):
    response = client.post("/v1/predict-fare", json={"pulocation_id": zone, "dolocation_id": 230})
    assert response.status_code == 422


def test_unknown_booking_type_rejected(client):
    payload = {"pulocation_id": 161, "dolocation_id": 230, "booking_type": "limo"}
    assert client.post("/v1/predict-fare", json=payload).status_code == 422


def test_returns_503_without_model():
    app.state.model = None
    response = TestClient(app).post("/v1/predict-fare", json={"pulocation_id": 161, "dolocation_id": 230})
    assert response.status_code == 503


def test_zones_endpoint(client):
    zones = client.get("/v1/zones").json()
    assert len(zones) == 263
    jfk = next(z for z in zones if z["id"] == 132)
    assert jfk["is_airport"] and jfk["borough"] == "Queens"
    assert next(z for z in zones if z["id"] == 230)["in_congestion_zone"]


def test_demo_page_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "/v1/predict-fare" in response.text
