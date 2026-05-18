import torch
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from datetime import datetime

from schema import SensorInput, PredictionOutput

from model_loader import (
    load_model,
    get_engine_status,
    scaler
)

# =========================================================
# LOAD MODEL
# =========================================================

model, feature_cols = load_model()

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="✈️ Aircraft Engine RUL Prediction API",
    description="""
Predictive Maintenance API for Aircraft Turbofan Engines
using NASA C-MAPSS dataset and LSTM Neural Networks.
""",
    version="2.1.0"
)

# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# =========================================================
# ROOT
# =========================================================

@app.get("/", tags=["System"])
def root():

    return {
        "message": "✈️ Aircraft Engine RUL Prediction API",
        "model": "LSTM v2.1",
        "status": "online",
        "docs": "/docs",
        "health": "/health"
    }

# =========================================================
# HEALTH
# =========================================================

@app.get("/health", tags=["System"])
@app.head("/health", tags=["System"])
def health_check():

    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "model": "LSTM v2.1"
    }

# =========================================================
# MODEL INFO
# =========================================================

@app.get("/model-info", tags=["System"])
def model_info():

    return {
        "model_name": "LSTM Neural Network",
        "version": "2.1.0",
        "dataset": "NASA C-MAPSS FD001",
        "val_rmse": 12.48,
        "val_mae": 9.14,
        "window_size": 30,
        "trained_on": "80 engines",
        "validated_on": "20 engines",
        "features_used": feature_cols,
        "normalization": "MinMaxScaler",
        "improvement_vs_rf": "+17%"
    }

# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Prediction"]
)
def predict_rul(data: SensorInput):

    try:

        # -------------------------------------------------
        # Convert request to dataframe
        # -------------------------------------------------

        input_dict = data.dict()

        input_df = pd.DataFrame([input_dict])

        # Ensure correct feature order
        input_df = input_df[feature_cols]

        # -------------------------------------------------
        # SCALE INPUT
        # -------------------------------------------------

        scaled_input = scaler.transform(input_df)

        # -------------------------------------------------
        # CONVERT TO TENSOR
        # -------------------------------------------------

        x = torch.tensor(
            scaled_input,
            dtype=torch.float32
        )

        # -------------------------------------------------
        # CREATE LSTM INPUT SHAPE
        # Final shape:
        # (1, 30, features)
        # -------------------------------------------------

        x = x.unsqueeze(0)

        x = x.repeat(1, 30, 1)

        # -------------------------------------------------
        # MODEL INFERENCE
        # -------------------------------------------------

        model.eval()

        with torch.no_grad():

            prediction = model(x)

            rul = float(prediction.item())

        # -------------------------------------------------
        # CLAMP OUTPUT
        # -------------------------------------------------

        rul = round(
            max(0, min(rul, 125)),
            2
        )

        # -------------------------------------------------
        # ENGINE STATUS
        # -------------------------------------------------

        status, confidence = get_engine_status(rul)

        # -------------------------------------------------
        # RETURN RESPONSE
        # -------------------------------------------------

        return PredictionOutput(
            predicted_rul=rul,
            status=status,
            confidence=confidence
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )