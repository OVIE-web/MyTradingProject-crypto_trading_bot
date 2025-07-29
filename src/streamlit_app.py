import streamlit as st
import pandas as pd
from src.model_manager import load_trained_model, make_predictions
from src.config import FEATURE_COLUMNS

model = load_trained_model()

st.title("Crypto Trading Bot Dashboard")

st.sidebar.header("Input Features")
user_input = {col: st.sidebar.number_input(col, value=0.0) for col in FEATURE_COLUMNS}

if st.button("Predict"):
    X = pd.DataFrame([user_input])[FEATURE_COLUMNS]
    preds, probs = make_predictions(model, X)
    st.write(f"Prediction: {int(preds[0])}, Confidence: {float(probs[0]):.2f}")