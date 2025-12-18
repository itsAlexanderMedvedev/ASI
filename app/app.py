import streamlit as st
from predict import predict
import json

st.set_page_config(page_title="Iris Predictor", page_icon="🌸")
st.title("Iris Predictor")
st.write("Enter the measurements and click **Predict**.")

sepal_len = st.number_input("Sepal length", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
sepal_wid = st.number_input("Sepal width",  min_value=0.0, max_value=10.0, value=3.5, step=0.1)
petal_len = st.number_input("Petal length", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
petal_wid = st.number_input("Petal width",  min_value=0.0, max_value=10.0, value=0.2, step=0.1)

if st.button("Predict"):
    result = predict([sepal_len, sepal_wid, petal_len, petal_wid])
    st.success(f"Predicted: **{result['class_name']}** (class {result['class_id']})")
    if result["proba"] is not None:
        st.write("Class probabilities:", [round(p, 3) for p in result["proba"]])

try:
    with open("app/model_meta.json") as f:
        meta = json.load(f)

    version = meta.get("version", "?")
    best_model = meta.get("best_model", "?")
    mlflow_run_id = meta.get("mlflow_run_id", "?")
    accuracy = meta.get("metrics", {}).get("accuracy", "?")

    mlflow_ui_link = f"http://127.0.0.1:5000/#/experiments?run_id={mlflow_run_id}"

    footer_text = f"Version: {version} • Best model: {best_model} • MLflow run: {mlflow_run_id} • Accuracy: {accuracy:.3f}"
    st.markdown("---")
    st.markdown(f"{footer_text} • [MLflow UI]({mlflow_ui_link})")

except FileNotFoundError:
    st.markdown("---")
    st.markdown("Something went wrong")