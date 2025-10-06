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


# 3️⃣ User Input Fields

col1, col2 = st.columns(2)
with col1:
    weather = st.selectbox("Weather", ["Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy", "Unknown"])
    traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Unknown"])
    vehicle = st.selectbox("Vehicle Type", ["Motorcycle", "Scooter", "Van", "Bicycle", "Other"])
    area = st.selectbox("Area", ["Urban", "Metropolitan", "Semi-Urban", "Other"])
    category = st.selectbox("Product Category", [
        "Clothing", "Electronics", "Sports", "Cosmetics", "Toys", "Snacks",
        "Apparel", "Jewelry", "Outdoors", "Grocery", "Books", "Kitchen", "Home", "Pet Supplies", "Skincare"
    ])
with col2:
    agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5)
    agent_age = st.slider("Agent Age", 18, 60, 30)
    distance_km = st.number_input("Distance (in km)", 0.0, 10000.0, 5.0)
    order_hour = st.slider("Order Hour (24-hour format)", 0, 23, 12)
    pickup_hour = st.slider("Pickup Hour (24-hour format)", 0, 23, 13)
    drop_latitude = st.number_input("Drop Latitude", 0.0, 50.0, 19.0)
    drop_longitude = st.number_input("Drop Longitude", 60.0, 100.0, 72.0)
    efficiency = st.slider("Delivery Agent Efficiency (0-1)", 0.0, 1.0, 0.75)

# Prepare input DataFrame with ALL expected columns
input_df = pd.DataFrame({
    "Weather": [weather],
    "Traffic": [traffic],
    "Vehicle": [vehicle],
    "Area": [area],
    "Category": [category],
    "Agent_Rating": [agent_rating],
    "Agent_Age": [agent_age],
    "Distance_km": [distance_km],
    "Order_Hour": [order_hour],
    "Pickup_Hour": [pickup_hour],
    "Drop_Latitude": [drop_latitude],
    "Drop_Longitude": [drop_longitude],
    "Efficiency": [efficiency]
})

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

