import streamlit as st
import pickle
import pandas as pd

# Load your trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Credit Risk Profitability Prediction")

st.write("""
Upload client data to predict credit risk and profitability.
""")

# Example: Define expected input features
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", min_value=0, value=50000)
debt = st.number_input("Current Debt", min_value=0, value=5000)
# Add as many fields as your model expects

if st.button("Predict"):
    # Create a DataFrame for a single sample
    input_data = pd.DataFrame([[age, income, debt]], columns=['age', 'income', 'debt'])
    prediction = model.predict(input_data)[0]
    st.write("Credit Risk Prediction:", "High" if prediction else "Low")

    # If your model predicts profitability or probability, add that here

st.write("Note: This is a demo app. Adjust the input fields as per your actual model features.")
