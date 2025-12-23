# ==============================
# DIABETES PREDICTION (SPYDER)
# ==============================

import pandas as pd
import pickle

# Load saved files
model = pickle.load(
    open(r"C:\Users\MADHURA\Downloads\trained_model.sav", "rb")
)

scaler = pickle.load(
    open(r"C:\Users\MADHURA\Downloads\scaler.sav", "rb")
)

columns = pickle.load(
    open(r"C:\Users\MADHURA\Downloads\columns.sav", "rb")
)

# Input data (ONE PERSON)
input_data = {
    "gender": "Female",
    "age": 55,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 28.0,
    "HbA1c_level": 6.0,
    "blood_glucose_level": 130
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# One-hot encoding
input_df = pd.get_dummies(
    input_df,
    columns=["gender", "smoking_history"],
    drop_first=True
)

# Add missing columns
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Ensure correct column order
input_df = input_df[columns]

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)

# Output
if prediction[0] == 0:
    print("✅ The person is NOT diabetic")
else:
    print("⚠️ The person IS diabetic")
