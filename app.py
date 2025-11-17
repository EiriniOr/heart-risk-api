from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Heart Disease Risk API")

# Load the trained heart model
model = joblib.load("model.pkl")

# Input the API expects
class Patient(BaseModel):
    age: int           # years
    sex: int           # 1 = male, 0 = female
    trestbps: float    # resting blood pressure
    chol: float        # cholesterol
    thalach: float     # max heart rate
    exang: int         # 1 = exercise-induced angina, 0 = no

@app.post("/predict")
def predict(patient: Patient):
    # Put patient data into a 2D array for the model
    x = np.array([[
        patient.age,
        patient.sex,
        patient.trestbps,
        patient.chol,
        patient.thalach,
        patient.exang,
    ]])

    # Model gives probability of heart disease
    prob_disease = model.predict_proba(x)[0, 1].item()
    label = "high_risk" if prob_disease >= 0.5 else "low_risk"

    return {
        "heart_disease_probability": prob_disease,
        "risk_label": label
    }