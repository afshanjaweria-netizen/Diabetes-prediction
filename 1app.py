import streamlit as st
import numpy as np
import pickle

# ===============================
# Load Model & Scaler
# ===============================
with open("diabetes_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ===============================
# App Title
# ===============================
st.set_page_config(page_title="Diabetes Prediction App")
st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details to predict diabetes")

# ===============================
# Input Fields
# (ORDER MUST MATCH TRAINING DATA)
# ===============================
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# ===============================
# Prediction
# ===============================
if st.button("Predict Diabetes"):
    # Create input array (VERY IMPORTANT ORDER)
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                             skin_thickness, insulin, bmi, dpf, age]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = rf_model.predict(input_scaled)[0]
    probability = rf_model.predict_proba(input_scaled)[0][1]

    # Display result
    st.write(f"### Prediction Probability: {probability:.2f}")

    if prediction == 1:
        st.error("⚠️ The patient is likely to have Diabetes")
    else:
        st.success("✅ The patient is unlikely to have Diabetes")

