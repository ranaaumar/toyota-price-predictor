import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Model load karein
with open('car_model.pkl', 'rb') as file:
    data = pickle.load(file)
    model = data["model"]
    columns = data["columns"]

st.title("🚗 Toyota Corolla Price Predictor")
st.write("Enter car details to get an estimated market price.")

# User Inputs
age = st.number_input("Car Age (Months)", min_value=1, max_value=100, value=24)
km = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000)
hp = st.number_input("Horse Power (HP)", min_value=60, max_value=200, value=90)
weight = st.number_input("Weight (KG)", min_value=1000, max_value=2000, value=1100)

if st.button("Predict Price"):
    # Input data ko model format mein laein
    input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
    input_df['Age_08_04'] = age
    input_df['KM'] = km
    input_df['HP'] = hp
    input_df['Weight'] = weight
    
    # Prediction
    prediction = model.predict(input_df)
    
    st.success(f"The estimated price is: €{prediction[0]:,.2f}")