
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model
model = joblib.load("car_price_model.pkl")

# Title
st.title("🚗 Car Price Prediction with Explainability")

# Input form
year = st.number_input("Year", 1990, 2025, 2015)
km_driven = st.number_input("Kilometers Driven", 0, 500000, 30000)
fuel = st.selectbox("Fuel Type", ['CNG', 'Diesel', 'LPG', 'Petrol', 'Electric'])
seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual', 'Trustmark Dealer'])
transmission = st.selectbox("Transmission", ['Automatic', 'Manual'])
owner = st.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])

# Encoding
fuel_map = {'CNG': 0, 'Diesel': 1, 'LPG': 2, 'Petrol': 3, 'Electric': 4}
seller_map = {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}
trans_map = {'Automatic': 0, 'Manual': 1}
owner_map = {
    'First Owner': 0,
    'Second Owner': 1,
    'Third Owner': 2,
    'Fourth & Above Owner': 3,
    'Test Drive Car': 4
}

input_data = pd.DataFrame([[year, km_driven,
                            fuel_map[fuel], seller_map[seller_type],
                            trans_map[transmission], owner_map[owner]]],
                          columns=['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner'])

# Predict
if st.button("Predict Price"):
    price = model.predict(input_data)[0]
    st.success(f"Predicted Car Price: ₹{int(price):,}")

    # Explain with SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    st.subheader("🔍 Feature Impact")
    fig = plt.figure()
    shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
    st.pyplot(fig)
