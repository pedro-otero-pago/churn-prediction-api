from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "..", "model.pkl"))
model_columns = joblib.load(os.path.join(BASE_DIR, "..", "model_columns.pkl"))

app = FastAPI()

class CustomerData(BaseModel):
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PaperlessBilling: str
    MonthlyCharges: float
    InternetService: str
    Contract: str
    PaymentMethod: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

@app.post("/predict")
def predict(customer: CustomerData):
    input_df = pd.DataFrame([customer.model_dump()])

    binary_columns = ["Partner", "Dependents", "PaperlessBilling"]
    for col in binary_columns:
        input_df[col] = input_df[col].map({"Yes": 1, "No": 0})

    onehot_columns = ["InternetService", "Contract", "PaymentMethod", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    input_df = pd.get_dummies(input_df, columns=onehot_columns)

    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "churn_prediction": bool(prediction),
        "churn_probability": round(float(probability), 4)
    }

@app.get("/health")
def health():
    return {"status": "ok"}