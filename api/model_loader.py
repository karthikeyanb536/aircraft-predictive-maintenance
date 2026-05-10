import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'rf_tuned.pkl')
FEATURE_PATH = os.path.join(BASE_DIR, 'models', 'feature_cols.pkl')

def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
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