# Heart Disease Risk Prediction API

> 📁 Part of my portfolio: [see this project and more →](https://eirini-portfolio-aer3.vercel.app/?utm_source=github&utm_medium=readme&utm_campaign=heart-risk-api#story/heart-disease-risk-prediction-api)

A simple end-to-end machine learning project that predicts the risk of heart disease using clinical features.  
The model is trained in Python using scikit-learn and served through a FastAPI application.

In this project:
- Machine learning workflow (training → saving → loading model)
- API development with FastAPI
- JSON validation with Pydantic
- Interactive API documentation (Swagger UI)
- Docker containerization (optional)
- Preparation for cloud deployment (Cloud Run / Azure / AWS)

---

## How the model works

The random forest classifier is trained on the publically available **UCI Heart Disease dataset**, using features such as:

- `age`
- `sex`
- `trestbps` – resting blood pressure
- `chol` – cholesterol
- `thalach` – maximum heart rate
- `exang` – exercise-induced angina

The output is:
- `heart_disease_probability` (0.0–1.0)
- `risk_label` (low_risk or high_risk)

---

## Local Development

### 1. Install dependencies:
```bash
pip install -r requirements.txt
