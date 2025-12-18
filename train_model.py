import os
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

EXPERIMENT_NAME = "iris-model-zoo"
mlflow.set_experiment(EXPERIMENT_NAME)

os.makedirs("app", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

best_f1 = -1
best_run_id = None
best_model = None
best_model_name = None
best_metrics = None

# ========================
# Training loop
# ========================
for name, clf in models.items():
    with mlflow.start_run(run_name=name):

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", clf)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        # ---- Metrics
        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        precision = precision_score(y_test, preds, average="macro")
        recall = recall_score(y_test, preds, average="macro")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # ROC-AUC (if available)
        try:
            proba = pipeline.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, proba, multi_class="ovr")
            mlflow.log_metric("roc_auc", roc_auc)
        except Exception:
            pass

        # ---- Params & tags
        mlflow.log_param("model_name", name)
        mlflow.set_tag("version", "v1.0.0")

        # ---- Confusion matrix
        cm = confusion_matrix(y_test, preds)
        plt.figure()
        plt.imshow(cm)
        plt.title(f"Confusion Matrix – {name}")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")

        cm_path = f"artifacts/cm_{name}.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # ---- Classification report
        report = classification_report(y_test, preds, target_names=iris.target_names)
        report_path = f"artifacts/report_{name}.txt"
        with open(report_path, "w") as f:
            f.write(report)

        mlflow.log_artifact(report_path)

        # ---- Save model
        model_path = f"artifacts/{name}.joblib"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(model_path)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="IrisModel"
        )

        # ---- Track best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline
            best_model_name = name
            best_run_id = mlflow.active_run().info.run_id
            best_metrics = {
                "accuracy": round(accuracy, 4),
                "f1_macro": round(f1, 4)
            }

# ========================
# Save best model locally
# ========================
joblib.dump(best_model, "app/model.joblib")

meta = {
    "best_model": best_model_name,
    "metrics": best_metrics,
    "mlflow_run_id": best_run_id,
    "version": "v1.0.0"
}

with open("app/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Best model:", best_model_name)
print("Saved to app/model.joblib")