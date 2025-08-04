import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
try:
    with open('parkinson_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Model or scaler files not found. Please run `train_model.py` first.")
    st.stop()

# --- Streamlit Web App Interface ---
st.set_page_config(page_title="Parkinson's Disease Detection", layout="wide")

# Title and description
st.title("Speech-Based Parkinson's Disease Detection Tool 🧠")
st.write("""
Enter the vocal measurements of a patient to predict if they have Parkinson's disease.
This tool uses a machine learning model trained on the UCI Parkinson's dataset.
""")

# Create columns for input fields for better layout
st.sidebar.header("Patient Vocal Features")
st.sidebar.write("Use the sliders to input feature values.")

def user_input_features():
    """Create sliders in the sidebar for user input."""
    # The keys should match the feature columns from the training data, in order
    features = {
        'MDVP:Fo(Hz)': st.sidebar.slider('MDVP:Fo(Hz)', 100.0, 300.0, 150.0),
        'MDVP:Fhi(Hz)': st.sidebar.slider('MDVP:Fhi(Hz)', 120.0, 600.0, 200.0),
        'MDVP:Flo(Hz)': st.sidebar.slider('MDVP:Flo(Hz)', 70.0, 250.0, 100.0),
        'MDVP:Jitter(%)': st.sidebar.slider('MDVP:Jitter(%)', 0.0, 0.05, 0.005),
        'MDVP:Jitter(Abs)': st.sidebar.slider('MDVP:Jitter(Abs)', 0.0, 0.0005, 0.00005),
        'MDVP:RAP': st.sidebar.slider('MDVP:RAP', 0.0, 0.03, 0.003),
        'MDVP:PPQ': st.sidebar.slider('MDVP:PPQ', 0.0, 0.03, 0.003),
        'Jitter:DDP': st.sidebar.slider('Jitter:DDP', 0.0, 0.09, 0.008),
        'MDVP:Shimmer': st.sidebar.slider('MDVP:Shimmer', 0.0, 0.15, 0.03),
        'MDVP:Shimmer(dB)': st.sidebar.slider('MDVP:Shimmer(dB)', 0.0, 1.5, 0.2),
        'Shimmer:APQ3': st.sidebar.slider('Shimmer:APQ3', 0.0, 0.08, 0.015),
        'Shimmer:APQ5': st.sidebar.slider('Shimmer:APQ5', 0.0, 0.1, 0.018),
        'MDVP:APQ': st.sidebar.slider('MDVP:APQ', 0.0, 0.15, 0.025),
        'Shimmer:DDA': st.sidebar.slider('Shimmer:DDA', 0.0, 0.25, 0.045),
        'NHR': st.sidebar.slider('NHR', 0.0, 0.5, 0.02),
        'HNR': st.sidebar.slider('HNR', 5.0, 40.0, 20.0),
        'RPDE': st.sidebar.slider('RPDE', 0.2, 0.8, 0.5),
        'DFA': st.sidebar.slider('DFA', 0.5, 0.9, 0.7),
        'spread1': st.sidebar.slider('spread1', -8.0, -2.0, -5.0),
        'spread2': st.sidebar.slider('spread2', 0.0, 0.5, 0.2),
        'D2': st.sidebar.slider('D2', 1.0, 4.0, 2.0),
        'PPE': st.sidebar.slider('PPE', 0.0, 0.6, 0.2)
    }
    data = pd.DataFrame(features, index=[0])
    return data

# Get user input
input_df = user_input_features()

# Main panel for displaying inputs and prediction
st.header("Patient's Input Features")
st.write(input_df)

# Prediction button
if st.sidebar.button("Predict"):
    # Scale the input features using the loaded scaler
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.header("Prediction Result")
    if prediction[0] == 1:
        st.error("The model predicts that this person has Parkinson's Disease.", icon="⚠️")
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.success("The model predicts that this person is Healthy.", icon="✅")
        st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")

