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

traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Unknown"])
distance = st.number_input("Distance (in km)", 0.0, 10000.0, 5.0)
agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5)


# 4️⃣ Prepare Input DataFrame

input_df = pd.DataFrame({
    "Traffic": [traffic],
    "Distance_km": [distance],
    "Agent_Rating": [agent_rating]
})

st.subheader("🧾 Input Summary")
st.dataframe(input_df)


# 5️⃣ Predict Delivery Time

if st.button("🔮 Predict Delivery Time"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🕒 Estimated Delivery Time: **{prediction:.2f} minutes**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# 6️⃣ Footer

st.markdown("---")
st.caption("Developed by **Aayush Kaul** | Machine Learning Delivery Prediction | Streamlit 🚀")
