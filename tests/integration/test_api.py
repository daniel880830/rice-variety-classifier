from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rice_classifier.api import create_app
from rice_classifier.config import Settings


def _client(model_path: Path, **overrides) -> TestClient:
    settings = Settings(_env_file=None, model_path=model_path, **overrides)
    return TestClient(create_app(settings))


@pytest.fixture
def client(model_path: Path):
    with _client(model_path, max_batch_size=3) as c:
        yield c


@pytest.fixture
def client_without_model(tmp_path: Path):
    with _client(tmp_path / "missing.joblib") as c:
        yield c


def test_root_redirects_to_docs(client: TestClient):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_health(client: TestClient):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client: TestClient):
    info = client.get("/model/info").json()
    assert "model_name" in info
    assert len(info["feature_names"]) == 7


def test_predict(client: TestClient, grain: dict):
    response = client.post("/predict", json=grain)
    assert response.status_code == 200
    body = response.json()
    assert body["class_name"] in {"Cammeo", "Osmancik"}
    assert body["label"] in {0, 1}
    assert 0.5 <= body["probability"] <= 1


def test_predict_accepts_null_measurement(client: TestClient, grain: dict):
    assert client.post("/predict", json={**grain, "extent": None}).status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        {"unexpected": 1},
        {"eccentricity": 2},
        {"area": 0},
        {"area": "big"},
        {"perimeter": [1, 2]},
    ],
)
def test_predict_rejects_invalid(client: TestClient, grain: dict, change: dict):
    assert client.post("/predict", json={**grain, **change}).status_code == 422


def test_predict_rejects_missing_field(client: TestClient, grain: dict):
    grain.pop("convex_area")
    assert client.post("/predict", json=grain).status_code == 422


def test_predict_rejects_non_json(client: TestClient):
    response = client.post("/predict", content="area=1", headers={"Content-Type": "text/plain"})
    assert response.status_code == 422


def test_batch(client: TestClient, grain: dict):
    response = client.post("/predict/batch", json={"items": [grain, grain]})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert predictions[0] == predictions[1]
    assert predictions[0] == client.post("/predict", json=grain).json()


def test_batch_empty(client: TestClient):
    assert client.post("/predict/batch", json={"items": []}).status_code == 422


def test_batch_too_large(client: TestClient, grain: dict):
    response = client.post("/predict/batch", json={"items": [grain] * 4})
    assert response.status_code == 413
    assert "limit of 3" in response.json()["detail"]


def test_batch_rejects_extra_top_level_field(client: TestClient, grain: dict):
    assert client.post("/predict/batch", json={"items": [grain], "x": 1}).status_code == 422


def test_without_model(client_without_model: TestClient, grain: dict):
    assert client_without_model.get("/health").json()["model_loaded"] is False
    assert client_without_model.get("/model/info").status_code == 503
    assert client_without_model.post("/predict", json=grain).status_code == 503
    assert client_without_model.post("/predict/batch", json={"items": [grain]}).status_code == 503


def test_corrupt_model_returns_503(tmp_path: Path, grain: dict):
    path = tmp_path / "corrupt.joblib"
    path.write_bytes(b"garbage")
    with _client(path) as c:
        assert c.post("/predict", json=grain).status_code == 503
