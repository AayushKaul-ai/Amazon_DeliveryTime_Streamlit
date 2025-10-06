import streamlit as st
import pandas as pd
import joblib

# Load your compressed model (must be in the same folder)

try:
    model = joblib.load("best_model_compressed.pkl")
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file not found. Make sure 'best_model_compressed.pkl' is in the same folder as app.py.")
    st.stop()

# 2️⃣ App Title and Description

st.set_page_config(page_title="Amazon Delivery Predictor", page_icon="🚚")
st.title("🚚 Amazon Delivery Time Prediction App")
st.write("Predict estimated delivery time based on traffic, distance, and agent performance details.")

st.subheader("🧾 Enter Delivery Details")

col1, col2 = st.columns(2)
with col1:
    agent_age = st.slider("Agent Age", 18, 60, 30)
    agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5)
    store_latitude = st.number_input("Store Latitude", 0.0, 50.0, 19.1)
    store_longitude = st.number_input("Store Longitude", 60.0, 100.0, 72.8)
    drop_latitude = st.number_input("Drop Latitude", 0.0, 50.0, 19.0)
with col2:
    drop_longitude = st.number_input("Drop Longitude", 60.0, 100.0, 73.0)
    distance_km = st.number_input("Distance (in km)", 0.0, 10000.0, 5.0)
    order_hour = st.slider("Order Hour (24-hour format)", 0, 23, 12)
    pickup_hour = st.slider("Pickup Hour (24-hour format)", 0, 23, 13)
    efficiency = st.slider("Agent Efficiency (0–1 scale)", 0.0, 1.0, 0.75)

# 4️⃣ Build DataFrame in the SAME ORDER as training

input_df = pd.DataFrame([[
    agent_age, agent_rating, store_latitude, store_longitude,
    drop_latitude, drop_longitude, distance_km,
    order_hour, pickup_hour, efficiency
]], columns=[
    'Agent_Age', 'Agent_Rating', 'Store_Latitude', 'Store_Longitude',
    'Drop_Latitude', 'Drop_Longitude', 'Distance_km',
    'Order_Hour', 'Pickup_Hour', 'Efficiency'
])
st.subheader("🧾 Input Summary")
st.dataframe(input_df)

# Predict
if st.button("🔮 Predict Delivery Time"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🕒 Estimated Delivery Time: **{prediction:.2f} minutes**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
   
# 6️⃣ Footer

st.markdown("---")
st.caption("Developed by **Aayush Kaul** | Machine Learning Delivery Prediction | Streamlit 🚀")

