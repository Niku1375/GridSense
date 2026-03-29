from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_prediction():
    payload = {
        "datetime": "2026-08-15 14:00:00",
        "region": "AEP"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_demand_MW" in response.json()