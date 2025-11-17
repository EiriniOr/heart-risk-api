from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Heart Disease Risk API")

# Load the trained model
model = joblib.load("model.pkl")

# Input schema
class Patient(BaseModel):
    age: int           # years
    sex: int           # 1 = male, 0 = female
    trestbps: float    # resting blood pressure
    chol: float        # cholesterol
    thalach: float     # max heart rate
    exang: int         # 1 = exercise-induced angina, 0 = no


@app.get("/", response_class=HTMLResponse)
def home():
    # Very simple HTML form + JS calling /predict
    return """
    <html>
      <head>
        <title>Heart Disease Risk Demo</title>
        <meta charset="utf-8" />
        <style>
          body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 0 16px; }
          label { display:block; margin-top: 10px; }
          input { width: 100%; padding: 6px; margin-top: 4px; }
          button { margin-top: 16px; padding: 8px 16px; cursor: pointer; }
          .result { margin-top: 20px; font-weight: bold; }
        </style>
      </head>
      <body>
        <h1>Heart Disease Risk – Demo</h1>
        <p>Enter some basic clinical values and get a risk estimate (demo only, not medical advice).</p>

        <label>Age (years)
          <input id="age" type="number" value="54" />
        </label>

        <label>Sex (1 = male, 0 = female)
          <input id="sex" type="number" value="1" />
        </label>

        <label>Resting blood pressure (trestbps)
          <input id="trestbps" type="number" value="130" />
        </label>

        <label>Cholesterol (chol)
          <input id="chol" type="number" value="250" />
        </label>

        <label>Max heart rate (thalach)
          <input id="thalach" type="number" value="150" />
        </label>

        <label>Exercise-induced angina (exang, 1 = yes, 0 = no)
          <input id="exang" type="number" value="0" />
        </label>

        <button onclick="sendRequest()">Predict risk</button>

        <div class="result" id="result"></div>

        <script>
          async function sendRequest() {
            const payload = {
              age: parseInt(document.getElementById('age').value),
              sex: parseInt(document.getElementById('sex').value),
              trestbps: parseFloat(document.getElementById('trestbps').value),
              chol: parseFloat(document.getElementById('chol').value),
              thalach: parseFloat(document.getElementById('thalach').value),
              exang: parseInt(document.getElementById('exang').value)
            };

            const resDiv = document.getElementById('result');
            resDiv.textContent = "Loading...";

            try {
              const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
              });
              const data = await response.json();
              resDiv.textContent = `Risk: ${ (data.heart_disease_probability * 100).toFixed(1) }% – ${data.risk_label}`;
            } catch (err) {
              resDiv.textContent = "Error calling API.";
            }
          }
        </script>
      </body>
    </html>
    """


@app.post("/predict")
def predict(patient: Patient):
    x = np.array([[
        patient.age,
        patient.sex,
        patient.trestbps,
        patient.chol,
        patient.thalach,
        patient.exang,
    ]])
    prob_disease = model.predict_proba(x)[0, 1].item()
    label = "high_risk" if prob_disease >= 0.5 else "low_risk"
    return {
        "heart_disease_probability": prob_disease,
        "risk_label": label
    }
