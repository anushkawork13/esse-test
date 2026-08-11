from fastapi.testclient import TestClient
from api.main import app

def test_predict():
    client = TestClient(app)
    response = client.post("/predict", json={"location": "sample", "property_type": "apartment", "area": 1200})
    assert response.status_code == 200
    assert "predicted_price" in response.json()
