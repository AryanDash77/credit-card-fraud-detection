import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load("fraud_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("CREDIT CARD FRAUD DETECTION")
st.write("Enter the transaction details in the sidebar to predict whether a transaction is legitimate or fraudulent.")

st.sidebar.header("Transaction Features")

amount_input = st.sidebar.text_input("Amount (€)", value = "100.0")
time_input = st.sidebar.text_input("Time (seconds)", value = "1.0")

try:
    amount = float(amount_input)
    time = float(time_input)

    if amount <= 0:
        st.sidebar.error("Amount must be greater than 0")
        amount = 100.0
    if time <= 0 :
        st.sidebar.error("Time must be greater than 0")
        time = 1.0
except:
    st.sidebar.error("Please enter valid numbers")
    amount = 100.0
    time = 1.0

v_features = []
for i in range(1,29):
    val = st.sidebar.slider(f"V{i}", min_value = -10.0, max_value = 10.0, value = 0.0, step = 0.1)
    v_features.append(val)


if st.button("Predict"):
    scaled_amount = scaler.transform([[amount]])[0][0]
    scaled_time = scaler.transform([[time]])[0][0]


    input_features = np.array(v_features + [scaled_amount, scaled_time]).reshape(1, -1)

    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]

    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED")
    else:
        st.success(f"✅ Legitimate Transaction")

    st.metric(label="Fraud Probability", value=f"{probability*100:.2f}%")



    