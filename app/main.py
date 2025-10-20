import os, joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.schema import DiabetesFeatures, PredictionResponse, HealthResponse

MODEL_PATH = os.getenv("MODEL_PATH", "model/model_v01.joblib")  # swap to v02 on release
MODEL_VERSION = os.getenv("MODEL_VERSION", "v0.1")

app = FastAPI(title="Virtual Diabetes Clinic Triage - ML Service")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle.get("trained_at", "")

pipeline, trained_at = load_model()

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_version=MODEL_VERSION)

@app.post("/predict", response_model=PredictionResponse)
def predict(features: DiabetesFeatures):
    try:
        X = [[features.age, features.sex, features.bmi, features.bp,
              features.s1, features.s2, features.s3, features.s4,
              features.s5, features.s6]]
        yhat = float(pipeline.predict(X)[0])
        return PredictionResponse(prediction=yhat)
    except Exception as e:
        # Observability: return JSON error
        raise HTTPException(status_code=400, detail={"error": str(e)})
