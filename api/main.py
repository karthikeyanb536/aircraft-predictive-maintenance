from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from datetime import datetime

from schema import SensorInput, PredictionOutput
from model_loader import load_model, get_engine_status

# Load model once at startup
model, feature_cols = load_model()

# Initialize FastAPI app
app = FastAPI(
    title="✈️ Aircraft Engine RUL Prediction API",
    description="Predictive Maintenance API for Aircraft Turbofan Engines using NASA C-MAPSS dataset",
    version="1.0.0"
)

# Allow frontend/dashboard to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─────────────────────────────────────────
# ENDPOINT 1: Health Check
# ─────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    return {
        "status"    : "online",
        "timestamp" : datetime.now().isoformat(),
        "model"     : "Random Forest v1.0"
    }

# ─────────────────────────────────────────
# ENDPOINT 2: Model Info
# ─────────────────────────────────────────
@app.get("/model-info", tags=["System"])
def model_info():
    return {
        "model_name"    : "Random Forest Regressor (tuned)",
        "version"       : "1.0.0",
        "val_rmse"      : 15.37,
        "val_mae"       : 10.94,
        "features_used" : feature_cols,
        "dataset"       : "NASA C-MAPSS FD001",
        "rul_cap"       : 125,
        "trained_on"    : "80 engines",
        "validated_on"  : "20 engines"
    }

# ─────────────────────────────────────────
# ENDPOINT 3: Predict RUL
# ─────────────────────────────────────────
@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_rul(data: SensorInput):
    try:
        # Convert input to dataframe
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])[feature_cols]

        # Predict
        rul = float(model.predict(input_df)[0])
        rul = round(max(0, min(rul, 125)), 2)

        # Get status
        status, confidence = get_engine_status(rul)

        return PredictionOutput(
            predicted_rul=rul,
            status=status,
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────
@app.get("/", tags=["System"])
def root():
    return {
        "message" : "✈️ Aircraft Engine RUL Prediction API",
        "docs"    : "/docs",
        "health"  : "/health",
        "predict" : "/predict"
    }