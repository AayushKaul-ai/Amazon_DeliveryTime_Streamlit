import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------
# Load your compressed model (must be in the same folder)
# ----------------------------------------------------------
model = joblib.load("best_model_compressed.pkl")

st.success("✅ Model loaded successfully!")

# ----------------------------------------------------------
# Example Streamlit UI 

st.title("🚚 Amazon Delivery Time Prediction App")
st.write("Predict delivery time based on traffic, weather, and agent details.")

# Example input (you should use your real columns here)
traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Unknown"])
distance = st.number_input("Distance (in km)", 0.0, 10000.0, 5.0)
agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5)

# Build input dataframe (match your training features)
input_df = pd.DataFrame({
    "Traffic": [traffic],
    "Distance_km": [distance],
    "Agent_Rating": [agent_rating]
})

if st.button("🔮 Predict Delivery Time"):
    prediction = model.predict(input_df)[0]
    st.success(f"🕒 Estimated Delivery Time: **{prediction:.2f} minutes**")
