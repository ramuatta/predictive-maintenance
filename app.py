
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="centered",
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
MODEL_INFO_PATH = BASE_DIR / "models" / "best_model_info.json"

FEATURE_COLUMNS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
]


@st.cache_resource
def load_model():
    """Load the trained model."""
    return joblib.load(MODEL_PATH)


st.title("⚙️ Predictive Maintenance")
st.write(
    "Enter the engine sensor readings below to predict "
    "the engine condition using the trained machine-learning model."
)

# Check that the saved model is available
if not MODEL_PATH.exists():
    st.error(
        "The trained model was not found. "
        "Please run the GitHub Actions training pipeline first."
    )
    st.stop()

# Display model information when available
if MODEL_INFO_PATH.exists():
    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as file:
        model_info = json.load(file)

    st.caption(
        f"Selected model: {model_info.get('selected_model', 'Not specified')}"
    )

# Sensor input fields
st.subheader("Engine Sensor Readings")

col1, col2 = st.columns(2)

with col1:
    engine_rpm = st.number_input(
        "Engine rpm",
        min_value=0.0,
        value=800.0,
        step=10.0,
    )
    lub_oil_pressure = st.number_input(
        "Lub oil pressure",
        min_value=0.0,
        value=3.0,
        step=0.1,
    )
    fuel_pressure = st.number_input(
        "Fuel pressure",
        min_value=0.0,
        value=6.0,
        step=0.1,
    )

with col2:
    coolant_pressure = st.number_input(
        "Coolant pressure",
        min_value=0.0,
        value=3.0,
        step=0.1,
    )
    lub_oil_temp = st.number_input(
        "Lub oil temp",
        min_value=0.0,
        value=80.0,
        step=0.5,
    )
    coolant_temp = st.number_input(
        "Coolant temp",
        min_value=0.0,
        value=80.0,
        step=0.5,
    )

# Predict the engine condition
if st.button("Predict Engine Condition", type="primary"):
    input_data = pd.DataFrame(
        [[
            engine_rpm,
            lub_oil_pressure,
            fuel_pressure,
            coolant_pressure,
            lub_oil_temp,
            coolant_temp,
        ]],
        columns=FEATURE_COLUMNS,
    )

    try:
        model = load_model()
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        st.metric("Predicted class", str(prediction))

        if prediction == 0:
            st.info(
                "The model predicted class 0. "
                "Review this result alongside sensor history "
                "and the applicable maintenance procedures."
            )
        elif prediction == 1:
            st.info(
                "The model predicted class 1. "
                "Review this result alongside sensor history "
                "and the applicable maintenance procedures."
            )
        else:
            st.warning("The model returned an unexpected class value.")

        st.caption(
            "This is a model prediction, not a confirmed diagnosis. "
            "Validate it against operational data before making "
            "maintenance decisions."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
