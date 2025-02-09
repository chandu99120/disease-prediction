import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Simulated dataset
data = {
    "fever": np.random.randint(0, 2, 100),
    "cough": np.random.randint(0, 2, 100),
    "fatigue": np.random.randint(0, 2, 100),
    "climate_factor": np.random.rand(100),
    "disease_label": np.random.randint(0, 2, 100),
}

df = pd.DataFrame(data)

# Split dataset
X = df.drop("disease_label", axis=1)
y = df["disease_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "disease_prediction_model.pkl")

