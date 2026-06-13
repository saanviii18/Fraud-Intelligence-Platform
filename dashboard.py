import streamlit as st
import joblib
import numpy as np

model = joblib.load("fraud_model.pkl")

st.set_page_config(
    page_title="Fraud Intelligence Platform",
    layout="wide"
)

st.title("💳 Fraud Intelligence Platform")

st.subheader("Real-Time Fraud Detection using XGBoost")

st.divider()

st.header("Transaction Prediction")

transaction = st.text_area(
    "Paste 30 transaction values separated by commas"
)

if st.button("Predict Fraud"):

    try:

        values = [float(x.strip()) for x in transaction.split(",")]

        arr = np.array(values).reshape(1, -1)

        prediction = model.predict(arr)[0]

        probability = model.predict_proba(arr)[0][1]

        if probability > 0.8:
            risk = "HIGH"
        elif probability > 0.4:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        if prediction == 1:
            st.error("🚨 Fraud Detected")
        else:
            st.success("✅ Normal Transaction")

        st.metric(
            "Fraud Probability",
            f"{probability*100:.2f}%"
            "Risk Level",
            risk
        )

    except Exception as e:
        st.error(f"Error: {e}")