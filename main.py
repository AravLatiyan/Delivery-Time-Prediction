import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🚚",
    layout="centered"
)

# ----------------------------------
# Load Dataset
# ----------------------------------
df = pd.read_csv("Food_Delivery_Times.csv")

# Drop rows with missing values
df = df.dropna()

# Convert categorical columns to string
categorical_cols = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type"
]

for col in categorical_cols:
    df[col] = df[col].astype(str)

# ----------------------------------
# Create Label Encoders
# ----------------------------------
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    le.fit(df[col])
    encoders[col] = le

# ----------------------------------
# Load Model
# ----------------------------------
with open("best_random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------------------
# Title
# ----------------------------------
st.title("🚚 Food Delivery Time Predictor")
st.write("Predict the estimated delivery time using Machine Learning.")

st.markdown("---")

# ----------------------------------
# User Inputs
# ----------------------------------
distance = st.number_input(
    "Distance (km)",
    min_value=0.0,
    value=5.0,
    step=0.5
)

weather = st.selectbox(
    "Weather",
    encoders["Weather"].classes_
)

traffic = st.selectbox(
    "Traffic Level",
    encoders["Traffic_Level"].classes_
)

time_of_day = st.selectbox(
    "Time of Day",
    encoders["Time_of_Day"].classes_
)

vehicle = st.selectbox(
    "Vehicle Type",
    encoders["Vehicle_Type"].classes_
)

prep_time = st.slider(
    "Preparation Time (minutes)",
    int(df["Preparation_Time_min"].min()),
    int(df["Preparation_Time_min"].max()),
    int(df["Preparation_Time_min"].median())
)

experience = st.slider(
    "Courier Experience (Years)",
    int(df["Courier_Experience_yrs"].min()),
    int(df["Courier_Experience_yrs"].max()),
    int(df["Courier_Experience_yrs"].median())
)

st.markdown("---")

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("🚀 Predict Delivery Time", use_container_width=True):

    weather_enc = encoders["Weather"].transform([weather])[0]
    traffic_enc = encoders["Traffic_Level"].transform([traffic])[0]
    tod_enc = encoders["Time_of_Day"].transform([time_of_day])[0]
    vehicle_enc = encoders["Vehicle_Type"].transform([vehicle])[0]

    input_data = pd.DataFrame({
        "Distance_km": [distance],
        "Weather": [weather_enc],
        "Traffic_Level": [traffic_enc],
        "Time_of_Day": [tod_enc],
        "Vehicle_Type": [vehicle_enc],
        "Preparation_Time_min": [prep_time],
        "Courier_Experience_yrs": [experience]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Delivery Time: {prediction:.2f} minutes")

    st.metric(
        label="Predicted Delivery Time",
        value=f"{prediction:.2f} min"
    )

st.markdown("---")
st.caption("Random Forest Regressor | Accuracy: 78%")