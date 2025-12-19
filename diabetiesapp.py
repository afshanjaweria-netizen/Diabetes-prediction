import streamlit as st
import numpy as np
import pickle
import os

# ======================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ======================================================
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# ======================================================
# SAFE MODEL LOADING (DEPLOYMENT-READY)
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

@st.cache_resource
def load_artifacts():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        return model, scaler

    except Exception as e:
        st.error("Model files could not be loaded.")
        st.stop()

rf_model, scaler = load_artifacts()

# ======================================================
# APP UI
# ======================================================
st.title("🩺 Diabetes Prediction System")
st.write("Enter patient medical details to assess diabetes risk.")

# ======================================================
# INPUT FIELDS
# ORDER MUST MATCH TRAINING DATA
# ======================================================
pregnancies = st.number_input("Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose Level", 0, 300, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin Level", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

# ======================================================
# PREDICTION
# ======================================================
if st.button("Predict Diabetes"):
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    try:
        input_scaled = scaler.transform(input_data)
        prediction = rf_model.predict(input_scaled)[0]
        probability = rf_model.predict_proba(input_scaled)[0][1]

        st.subheader(f"Prediction Probability: {probability:.2f}")

        if prediction == 1:
            st.error("⚠️ High likelihood of Diabetes detected")
        else:
            st.success("✅ Low likelihood of Diabetes detected")

    except Exception as e:
        st.error("Prediction failed. Input format mismatch.")
