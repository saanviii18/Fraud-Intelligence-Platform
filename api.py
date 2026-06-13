from fastapi import FastAPI, Body
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("fraud_model.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}


@app.post("/predict")
def predict(transaction: list = Body(...)):

    arr = np.array(transaction).reshape(1, -1)

    prediction = model.predict(arr)[0]

    probability = model.predict_proba(arr)[0][1]

    risk = "LOW"

    if probability > 0.8:
        risk = "HIGH"
    elif probability > 0.4:
        risk = "MEDIUM"

    return {
        "prediction": "Fraud" if prediction == 1 else "Normal",
        "fraud_probability": round(float(probability), 4),
        "risk_level": risk
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }