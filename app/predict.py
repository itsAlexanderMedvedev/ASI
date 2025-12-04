import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
artifact = joblib.load(MODEL_PATH)
_model = artifact["model"]
NAMES = artifact["target_names"]

def predict(features):
    X = [list(map(float, features))]
    y = _model.predict(X)[0]
    try:
        proba = _model.predict_proba(X)[0].tolist()
    except Exception:
        proba = None
    return {"class_id": int(y), "class_name": NAMES[int(y)], "proba": proba}