import joblib
import torch
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'lstm_tuned.pt')
FEATURE_PATH = os.path.join(BASE_DIR, 'models', 'feature_cols.pkl')

def load_model():
    from src.models.lstm_model import LSTMModelV2
    features = joblib.load(FEATURE_PATH)
    model = LSTMModelV2(input_size=len(features))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model, features

def get_engine_status(rul: float) -> tuple:
    if rul <= 30:
        return "🔴 CRITICAL", "High"
    elif rul <= 60:
        return "🟡 WARNING", "High"
    elif rul <= 100:
        return "🟢 MODERATE", "Medium"
    else:
        return "✅ HEALTHY", "Medium"