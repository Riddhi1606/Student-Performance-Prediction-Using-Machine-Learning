import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("xgboost_student_model.pkl")

st.title("Student Performance Prediction")

st.write("Enter student details to predict final marks")

gender = st.selectbox("Gender", [0, 1])
school_type = st.selectbox("School Type", [0, 1])

study_hours_daily = st.number_input(
    "Study Hours Daily",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

attendance_pct = st.number_input(
    "Attendance %",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

previous_score = st.number_input(
    "Previous Score",
    min_value=0,
    max_value=100,
    value=50
)

extracurricular = st.selectbox(
    "Extracurricular Activities",
    [0, 1]
)

parent_education = st.selectbox(
    "Parent Education",
    [0, 1, 2]
)

study_effectiveness = st.number_input(
    "Study Effectiveness",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

sleep_study_ratio = st.number_input(
    "Sleep Study Ratio",
    min_value=0.0,
    max_value=5.0,
    value=1.0
)

engagement_score = st.number_input(
    "Engagement Score",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

if st.button("Predict Final Marks"):

    data = pd.DataFrame({
        "gender": [gender],
        "school_type": [school_type],
        "study_hours_daily": [study_hours_daily],
        "attendance_pct": [attendance_pct],
        "sleep_hours": [sleep_hours],
        "previous_score": [previous_score],
        "extracurricular": [extracurricular],
        "parent_education": [parent_education],
        "study_effectiveness": [study_effectiveness],
        "sleep_study_ratio": [sleep_study_ratio],
        "engagement_score": [engagement_score]
    })

    prediction = model.predict(data)

    st.success(
        f"Predicted Final Marks: {prediction[0]:.2f}"
    )
