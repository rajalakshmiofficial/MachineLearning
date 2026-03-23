import streamlit as st
import pandas as pd
import joblib

# Load saved pipeline
pipeline = joblib.load("insurance_pipeline.pkl")

# Streamlit UI
st.title("💰 Insurance Charges Prediction App")

st.write("Enter customer details below to predict medical insurance charges.")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Create DataFrame for prediction
input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],          # raw string
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],    # raw string
    "region": [region]     # raw string
})

# Prediction Button
if st.button("🔮 Predict Charges"):
    prediction = pipeline.predict(input_df)[0]
    st.success(f"✅ Predicted Insurance Charge: **${prediction:,.2f}**")
