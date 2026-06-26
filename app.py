import streamlit as st
import pandas as pd
import joblib

st.sidebar.title("About")

st.sidebar.info(
"""
Student Performance Prediction

Model Used:
• XGBoost Regressor

Developed using:
• Python
• Streamlit
• Scikit-learn
• XGBoost
"""
)

st.sidebar.title("📌 Project Information")

st.sidebar.write("""
### Machine Learning Model

- XGBoost Regressor

### Dataset Features

- Academic
- Attendance
- Sleep
- Engagement
- Parent Education

### Technologies

- Python
- Streamlit
- Scikit-learn
- XGBoost
""")

with st.expander("📖 About this Project"):

    st.write("""
This project predicts student academic performance
using Machine Learning.

The model was trained using student-related
features like attendance, study hours,
previous score and engagement.
""")

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction System")

st.markdown("""
### Predict a student's final marks using Machine Learning

This application uses an **XGBoost Regressor** trained on student academic and behavioral data to estimate a student's final marks.

👨‍💻 Developed using **Python, Streamlit, Scikit-learn and XGBoost**
""")

col1, col2 = st.columns(2)

with col1:
    gender_option = st.selectbox("Gender", ["Male", "Female"])
    gender = 0 if gender_option == "Male" else 1

    school_option = st.selectbox("School Type", ["Public", "Private"])
    school_type = 0 if school_option == "Public" else 1

    study_hours_daily = st.number_input("Study Hours Daily", 0.0, 1.0, 0.5)

    attendance_pct = st.number_input("Attendance %", 0.0, 1.0, 0.5)

    sleep_hours = st.number_input("Sleep Hours", 0.0, 1.0, 0.5)

with col2:
    previous_score = st.number_input("Previous Score", 0, 100, 50)

    extra_option = st.selectbox(
        "Participates in Extracurricular Activities?",
        ["No", "Yes"]
    )
    extracurricular = 0 if extra_option == "No" else 1

    parent_option = st.selectbox(
        "Parent Education",
        ["High School", "Graduate", "Post Graduate"]
    )

    parent_map = {
        "High School": 0,
        "Graduate": 1,
        "Post Graduate": 2
    }

    parent_education = parent_map[parent_option]

    study_effectiveness = st.number_input("Study Effectiveness", 0.0, 1.0, 0.5)

    sleep_study_ratio = st.number_input("Sleep Study Ratio", 0.0, 5.0, 1.0)

    engagement_score = st.number_input("Engagement Score", 0.0, 1.0, 0.5)

# Load model
model = joblib.load("xgboost_student_model.pkl")

st.title("🎓 Student Performance Prediction System")

st.write("Enter student details to predict final marks")

st.write(
    """
    This application predicts student final marks using an XGBoost Machine Learning model.
    Enter the student's details below and click **Predict Final Marks**.
    """
)

gender_option = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

gender = 0 if gender_option == "Male" else 1

school_option = st.selectbox(
    "School Type",
    ["Public", "Private"]
)

school_type = 0 if school_option == "Public" else 1

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

extra_option = st.selectbox(
    "Participates in Extracurricular Activities?",
    ["No", "Yes"]
)

extracurricular = 0 if extra_option == "No" else 1

parent_option = st.selectbox(
    "Parent Education",
    ["High School", "Graduate", "Post Graduate"]
)

parent_map = {
    "High School":0,
    "Graduate":1,
    "Post Graduate":2
}

parent_education = parent_map[parent_option]

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
    f"🎯 Predicted Final Marks: {prediction[0]:.2f}"
)

st.balloons()

st.success(f"🎯 Predicted Final Marks: {prediction[0]:.2f}/100")

st.progress(min(int(prediction[0]), 100))

st.markdown("---")

st.caption(
    "Developed by Team Bug Slayers | AI & Machine Learning Project"
)

if st.button("🚀 Predict Final Marks"):

    with st.spinner("Predicting..."):

        prediction = model.predict(data)

    st.success(
        f"🎯 Predicted Final Marks: {prediction[0]:.2f}"
    )

st.subheader("Student Details")

st.write(data)

