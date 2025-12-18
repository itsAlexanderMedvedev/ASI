import os
import joblib
import os
from sklearn.datasets import load_iris

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
_model = joblib.load(MODEL_PATH)

# Use scikit-learn's Iris target names
iris = load_iris()
NAMES = iris.target_names.tolist()

def predict(features):
    X = [list(map(float, features))]
    y = _model.predict(X)[0]
    try:
        proba = _model.predict_proba(X)[0].tolist()
    except Exception:
        proba = None
    return {"class_id": int(y), "class_name": NAMES[int(y)], "proba": proba}