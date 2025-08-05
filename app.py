import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

# --- Load the Model and Scaler ---
try:
    # Initialize a new XGBoost model
    model = XGBClassifier()
    # Load the data from the JSON file
    model.load_model('parkinson_model.json')

    # Load the scaler from the pickle file
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

except FileNotFoundError:
    st.error("Model files not found. Please run `train_model.py` first to create them.")
    st.stop()


# --- Streamlit App Interface ---
st.set_page_config(page_title="Parkinson's Disease Detection", layout="wide")
st.title("Speech-Based Parkinson's Disease Detection Tool 🧠")
st.write("Enter the patient's vocal measurements to predict Parkinson's disease.")

st.sidebar.header("Patient Vocal Features")

def user_input_features():
    # This function creates the sliders for user input
    features = {
        'MDVP:Fo(Hz)': st.sidebar.slider('MDVP:Fo(Hz)', 88.0, 261.0, 150.0),
        'MDVP:Fhi(Hz)': st.sidebar.slider('MDVP:Fhi(Hz)', 102.0, 593.0, 200.0),
        'MDVP:Flo(Hz)': st.sidebar.slider('MDVP:Flo(Hz)', 65.0, 240.0, 100.0),
        'MDVP:Jitter(%)': st.sidebar.slider('MDVP:Jitter(%)', 0.001, 0.035, 0.005),
        'MDVP:Jitter(Abs)': st.sidebar.slider('MDVP:Jitter(Abs)', 0.00001, 0.00027, 0.00005),
        'MDVP:RAP': st.sidebar.slider('MDVP:RAP', 0.0006, 0.022, 0.003),
        'MDVP:PPQ': st.sidebar.slider('MDVP:PPQ', 0.0009, 0.02, 0.003),
        'Jitter:DDP': st.sidebar.slider('Jitter:DDP', 0.002, 0.065, 0.008),
        'MDVP:Shimmer': st.sidebar.slider('MDVP:Shimmer', 0.009, 0.12, 0.03),
        'MDVP:Shimmer(dB)': st.sidebar.slider('MDVP:Shimmer(dB)', 0.08, 1.31, 0.2),
        'Shimmer:APQ3': st.sidebar.slider('Shimmer:APQ3', 0.004, 0.056, 0.015),
        'Shimmer:APQ5': st.sidebar.slider('Shimmer:APQ5', 0.005, 0.069, 0.018),
        'MDVP:APQ': st.sidebar.slider('MDVP:APQ', 0.007, 0.14, 0.025),
        'Shimmer:DDA': st.sidebar.slider('Shimmer:DDA', 0.013, 0.17, 0.045),
        'NHR': st.sidebar.slider('NHR', 0.0006, 0.32, 0.02),
        'HNR': st.sidebar.slider('HNR', 8.0, 34.0, 20.0),
        'RPDE': st.sidebar.slider('RPDE', 0.25, 0.69, 0.5),
        'DFA': st.sidebar.slider('DFA', 0.57, 0.83, 0.7),
        'spread1': st.sidebar.slider('spread1', -7.97, -2.43, -5.0),
        'spread2': st.sidebar.slider('spread2', 0.0, 0.45, 0.2),
        'D2': st.sidebar.slider('D2', 1.42, 3.68, 2.0),
        'PPE': st.sidebar.slider('PPE', 0.04, 0.53, 0.2)
    }
    data = pd.DataFrame(features, index=[0])
    return data

input_df = user_input_features()

st.header("Patient's Input Features")
st.write(input_df)

if st.sidebar.button("Predict"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    st.header("Prediction Result")
    if prediction[0] == 1:
        st.error("The model predicts this person has Parkinson's Disease.", icon="⚠️")
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.success("The model predicts this person is Healthy.", icon="✅")
        st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")
