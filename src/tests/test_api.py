from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

valid_customer = {
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PaperlessBilling": "Yes",
    "MonthlyCharges": 89.5,
    "InternetService": "Fiber optic",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes"
}

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_returns_valid_response(valid_customer=valid_customer):
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert isinstance(data["churn_prediction"], bool)
    assert 0 <= data["churn_probability"] <= 1

def test_predict_rejects_invalid_data_type():
    invalid_customer = valid_customer.copy()
    invalid_customer["tenure"] = "not a number"
    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422

def test_predict_rejects_missing_field():
    incomplete_customer = valid_customer.copy()
    del incomplete_customer["Contract"]
    response = client.post("/predict", json=incomplete_customer)
    assert response.status_code == 422