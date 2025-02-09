from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained ML model
model = joblib.load("disease_prediction_model.pkl")

@app.route("/")
def home():
    return "Welcome to the Disease Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  
    features = np.array([data["fever"], data["cough"], data["fatigue"], data["climate_factor"]])
    prediction = model.predict([features])[0]
    
    return jsonify({"prediction": "Outbreak" if prediction == 1 else "No Outbreak"})

if __name__ == "__main__":
    app.run(debug=True)

