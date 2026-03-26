import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Toyota Corolla Price Predictor", page_icon="🚗")

# --- MODEL LOADING ---
# Path handling for Streamlit Cloud
curr_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(curr_path, 'car_model.sav')

@st.cache_resource # Taake model bar-bar load na ho
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

data = load_model()

if data is None:
    st.error("Error: 'car_model.sav' file nahi mili. Please check your GitHub repository.")
else:
    model = data["model"]
    columns = data["columns"]

    # --- UI DESIGN ---
    st.title("🚗 Toyota Corolla Price Predictor")
    st.write("Enter the vehicle details below to estimate its market value in Euros.")
    
    st.divider()

    # --- USER INPUTS ---
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age of the Car (Months)", min_value=1, max_value=100, value=24)
        km = st.number_input("Total Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000)
    
    with col2:
        hp = st.number_input("Horse Power (HP)", min_value=60, max_value=200, value=90)
        weight = st.number_input("Weight of the Car (KG)", min_value=1000, max_value=2000, value=1100)

    # --- PREDICTION LOGIC ---
    if st.button("Calculate Estimated Price", type="primary"):
        # Create a DataFrame with all columns initialized to 0
        input_data = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
        
        # Fill the input values
        input_data['Age_08_04'] = age
        input_data['KM'] = km
        input_data['HP'] = hp
        input_data['Weight'] = weight
        
        # Make Prediction
        prediction = model.predict(input_data)
        final_price = round(prediction[0], 2)

        # Display Result
        st.balloons()
        st.success(f"### Estimated Market Price: €{final_price:,}")
        
        # Context for the user
        st.info("Note: This prediction is based on the trained Linear Regression model (R²: 0.8865).")

st.divider()
st.caption("Developed for Machine Learning Assignment | MSDS-1")
