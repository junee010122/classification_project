import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.title("Classifier Dashboard")

st.header("Backend Status")
try:
    health = requests.get(f"{API_URL}/").json()
    st.success(health.get("message", "No message returned"))
except Exception as e:
    st.error(f"Could not connect to FastAPI: {e}")

st.header("Model Metrics")
try:
    metrics = requests.get(f"{API_URL}/metrics").json()
    if "error" in metrics:
        st.error(metrics["error"])
    else:
        st.json(metrics)
except Exception as e:
    st.error(f"Could not load metrics: {e}")

st.header("Test Predictions")
try:
    preds = requests.get(f"{API_URL}/predictions").json()
    if isinstance(preds, dict) and "error" in preds:
        st.error(preds["error"])
    else:
        preds_df = pd.DataFrame(preds)
        st.dataframe(preds_df)
except Exception as e:
    st.error(f"Could not load predictions: {e}")

st.header("Make a New Prediction")
try:
    if 'preds_df' in locals() and not preds_df.empty:
        feature_cols = [c for c in preds_df.columns if c.startswith("feature_")]
    else:
        feature_cols = [f"feature_{i}" for i in range(590)]

    user_input = {}
    with st.form(key="predict_form"):
        for col in feature_cols:
            user_input[col] = st.number_input(col, value=0.0)
        submit = st.form_submit_button("Predict")

    if submit:
        result = requests.post(f"{API_URL}/predict", json=user_input).json()
        if "error" in result:
            st.error(result["error"])
        else:
            from IPython.core.debugger import set_trace
            set_trace()
            st.success(f"Prediction: {result['prediction']}")
            st.info(f"Probability: {result['probability']:.4f}")

except Exception as e:
    st.error(f"Could not load prediction form: {e}")
