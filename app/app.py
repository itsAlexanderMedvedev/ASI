import streamlit as st
from predict import predict

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