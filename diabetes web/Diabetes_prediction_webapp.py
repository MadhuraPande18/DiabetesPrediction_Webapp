# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 23:15:01 2025

@author: Madhura
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model, scaler, and columns
loaded_model = pickle.load(open(r"C:/Users/MADHURA/Downloads/trained_model.sav", "rb"))
scaler = pickle.load(open(r"C:/Users/MADHURA/Downloads/scaler.sav", "rb"))
columns = pickle.load(open(r"C:/Users/MADHURA/Downloads/columns.sav", "rb"))

# creating a function for Prediction
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # scale the input
    input_data_scaled = scaler.transform(input_data_reshaped)

    prediction = loaded_model.predict(input_data_scaled)

    if prediction[0] == 0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"


def main():
    # giving a title
    st.title("Diabetes Prediction Web App")

    # getting the input data from the user
    age = st.text_input("Age")
    hypertension = st.text_input("Hypertension (0 = No, 1 = Yes)")
    heart_disease = st.text_input("Heart Disease (0 = No, 1 = Yes)")
    bmi = st.text_input("BMI value")
    HbA1c_level = st.text_input("HbA1c Level")
    blood_glucose_level = st.text_input("Blood Glucose Level")

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    smoking_history = st.selectbox(
        "Smoking History",
        ["current", "ever", "former", "never", "not current"]
    )

    # one-hot encoding for gender
    gender_Male = 1 if gender == "Male" else 0
    gender_Other = 1 if gender == "Other" else 0

    # one-hot encoding for smoking history
    smoking_current = 1 if smoking_history == "current" else 0
    smoking_ever = 1 if smoking_history == "ever" else 0
    smoking_former = 1 if smoking_history == "former" else 0
    smoking_never = 1 if smoking_history == "never" else 0
    smoking_not_current = 1 if smoking_history == "not current" else 0

    # code for Prediction
    diagnosis = ""

    # creating a button for Prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([
            age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level,
            gender_Male, gender_Other,
            smoking_current, smoking_ever, smoking_former, smoking_never, smoking_not_current
        ])

    st.success(diagnosis)


if __name__ == "__main__":
    main()
