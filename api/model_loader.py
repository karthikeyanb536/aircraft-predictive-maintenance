import joblib
import torch
import os
import sys

# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(BASE_DIR)

# =========================================================
# PATHS
# =========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "lstm_tuned.pt"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_cols.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "minmax_scaler.pkl"
)

# =========================================================
# LOAD SCALER
# =========================================================

scaler = joblib.load(SCALER_PATH)

# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    from src.models.lstm_model import LSTMModelV2

    features = joblib.load(FEATURE_PATH)

    model = LSTMModelV2(
        input_size=len(features)
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    return model, features

# =========================================================
# ENGINE STATUS
# =========================================================

def get_engine_status(rul: float):

    if rul <= 30:
        return "🔴 CRITICAL", "High"

    elif rul <= 60:
        return "🟡 WARNING", "High"

    elif rul <= 100:
        return "🟢 MODERATE", "Medium"

    else:
        return "✅ HEALTHY", "Medium"