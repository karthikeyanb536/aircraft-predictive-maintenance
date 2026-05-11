import torch
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from schema import SensorInput, PredictionOutput
from model_loader import load_model, get_engine_status

# Load model once at startup
model, feature_cols = load_model()

app = FastAPI(
    title="✈️ Aircraft Engine RUL Prediction API",
    description="Predictive Maintenance API for Aircraft Turbofan Engines using NASA C-MAPSS dataset",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", tags=["System"])
def root():
    return {
        "message" : "✈️ Aircraft Engine RUL Prediction API v2.0",
        "model"   : "LSTM Neural Network",
        "docs"    : "/docs",
        "health"  : "/health",
        "predict" : "/predict"
    }

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status"    : "online",
        "timestamp" : datetime.now().isoformat(),
        "model"     : "LSTM v2.0"
    }

@app.get("/model-info", tags=["System"])
def model_info():
    return {
        "model_name"    : "LSTM Neural Network (tuned)",
        "version"       : "2.0.0",
        "val_rmse"      : 12.48,
        "val_mae"       : 9.14,
        "features_used" : feature_cols,
        "dataset"       : "NASA C-MAPSS FD001",
        "rul_cap"       : 125,
        "trained_on"    : "80 engines",
        "validated_on"  : "20 engines",
        "improvement"   : "17% better than Random Forest v1.0"
    }

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_rul(data: SensorInput):
    try:
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])[feature_cols]

        # LSTM needs (batch, sequence, features)
        x = torch.tensor(input_df.values, dtype=torch.float32)
        x = x.unsqueeze(0).repeat(1, 30, 1)

        with torch.no_grad():
            rul = float(model(x).item())

        rul = round(max(0, min(rul, 125)), 2)
        status, confidence = get_engine_status(rul)

        return PredictionOutput(
            predicted_rul=rul,
            status=status,
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))