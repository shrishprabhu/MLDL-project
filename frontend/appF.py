# streamlit frontend for HHAR Activity Recognition API

import streamlit as st
import requests

st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="centered"
)

st.title("🏃 Human Activity Recognition")
st.write("Predict human activity using accelerometer data.")

st.divider()

x = st.number_input("Accelerometer X", value=0.0)
y = st.number_input("Accelerometer Y", value=0.0)
z = st.number_input("Accelerometer Z", value=0.0)

if st.button("Predict Activity", use_container_width=True):

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={
                "x": x,
                "y": y,
                "z": z
            }
        )

        if response.status_code == 200:
            activity = response.json()["activity"]

            st.success(f"Predicted Activity: **{activity.upper()}**")
        else:
            st.error("Prediction failed.")

    except Exception as e:
        st.error(f"Cannot connect to backend.\n\n{e}")