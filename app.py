import streamlit as st
import pandas as pd
import pickle

# --- 1. Load saved model and scaler ---
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Heart Disease Prediction App")

st.write("Enter patient details below:")

# --- 2. User input ---
age = st.number_input("Age", min_value=1, max_value=120, value=45)
resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=130)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=250)
max_heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.5)

input_data = pd.DataFrame({
    'age': [age],
    'resting_bp': [resting_bp],
    'cholesterol': [cholesterol],
    'max_heart_rate': [max_heart_rate],
    'oldpeak': [oldpeak],
    # Add other columns here
})

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)[0][1]

st.subheader("Prediction Result")
if prediction[0] == 1:
    st.error(f"The patient is likely to have Heart Disease. (Probability: {prediction_proba:.2f})")
else:
    st.success(f"The patient is unlikely to have Heart Disease. (Probability: {prediction_proba:.2f})")
