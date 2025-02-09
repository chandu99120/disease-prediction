import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("disease_prediction_model.pkl")

st.title("Disease Outbreak Prediction")

fever = st.slider("Fever (0 - No, 1 - Yes)", 0, 1, 0)
cough = st.slider("Cough (0 - No, 1 - Yes)", 0, 1, 0)
fatigue = st.slider("Fatigue (0 - No, 1 - Yes)", 0, 1, 0)
climate_factor = st.slider("Climate Factor (0 to 1)", 0.0, 1.0, 0.5)

if st.button("Predict"):
    features = np.array([[fever, cough, fatigue, climate_factor]])
    prediction = model.predict(features)[0]
    result = "Outbreak" if prediction == 1 else "No Outbreak"
    st.write(f"Prediction: **{result}**")

