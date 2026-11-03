
import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib
import numpy as np

st.set_page_config(page_title="DeepShield", page_icon="🛡️", layout="wide")

MODEL_PATH = "/content/drive/MyDrive/Credit Card Fraud Detection System/models/deepshield_model.keras"
SCALER_PATH = "/content/drive/MyDrive/Credit Card Fraud Detection System/models/scaler.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.title("🛡️ DeepShield")
st.subheader("Credit Card Fraud Detection Using Deep Learning")
st.write("Upload a CSV file containing transaction data to detect potentially fraudulent transactions.")

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    required_columns = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
    else:
        features = data[required_columns].copy()
        features[["Time", "Amount"]] = scaler.transform(features[["Time", "Amount"]])

        probabilities = model.predict(features, verbose=0).ravel()
        predictions = (probabilities >= 0.95).astype(int)

        results = data.copy()
        results["Fraud Probability"] = probabilities
        results["Prediction"] = np.where(predictions == 1, "Fraud", "Genuine")

        fraud_count = int(predictions.sum())
        col1, col2, col3 = st.columns(3)
        col1.metric("Transactions", len(results))
        col2.metric("Potential Fraud", fraud_count)
        col3.metric("Genuine", len(results) - fraud_count)

        st.dataframe(results, use_container_width=True)

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions",
            data=csv,
            file_name="deepshield_predictions.csv",
            mime="text/csv"
        )
