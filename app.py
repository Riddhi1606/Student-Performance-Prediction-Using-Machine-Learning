import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("xgboost_student_model.pkl")

# ---------------- HEADER ----------------
st.title("🎓 Student Performance Prediction System")

st.markdown("""
This application predicts **final marks of students** using a trained **XGBoost Machine Learning model**.

Fill in the details below and click **Predict**.
""")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 About Project")
st.sidebar.info("""
- ML Model: XGBoost Regressor  
- Task: Regression (Final Marks Prediction)  
- Dataset: Student Performance Dataset  
- Built using: Streamlit + Python  
""")

# ---------------- INPUT UI ----------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 0 if gender == "Male" else 1

    school_type = st.selectbox("School Type", ["Public", "Private"])
    school_type = 0 if school_type == "Public" else 1

    study_hours_daily = st.number_input("Study Hours Daily", 0.0, 1.0, 0.5)
    attendance_pct = st.number_input("Attendance %", 0.0, 1.0, 0.5)
    sleep_hours = st.number_input("Sleep Hours", 0.0, 1.0, 0.5)

with col2:
    previous_score = st.number_input("Previous Score", 0, 100, 50)

    extracurricular = st.selectbox("Extracurricular", ["No", "Yes"])
    extracurricular = 0 if extracurricular == "No" else 1

    parent_education = st.selectbox(
        "Parent Education",
        ["High School", "Graduate", "Post Graduate"]
    )

    parent_map = {
        "High School": 0,
        "Graduate": 1,
        "Post Graduate": 2
    }
    parent_education = parent_map[parent_education]

    study_effectiveness = st.number_input("Study Effectiveness", 0.0, 1.0, 0.5)
    sleep_study_ratio = st.number_input("Sleep Study Ratio", 0.0, 5.0, 1.0)
    engagement_score = st.number_input("Engagement Score", 0.0, 1.0, 0.5)

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Final Marks"):

    input_data = pd.DataFrame({
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

    with st.spinner("Predicting result..."):
        prediction = model.predict(input_data)[0]

    prediction = float(prediction)

    # ---------------- RESULT ----------------
    st.success(f"🎯 Predicted Final Marks: {prediction:.2f}/100")

    st.progress(min(int(prediction), 100))

    # ---------------- PERFORMANCE CATEGORY ----------------
    if prediction >= 90:
        st.success("🏆 Performance: Excellent")
    elif prediction >= 75:
        st.info("🌟 Performance: Good")
    elif prediction >= 50:
        st.warning("📚 Performance: Average")
    else:
        st.error("⚠️ Performance: Needs Improvement")

    # ---------------- DETAILS ----------------
    with st.expander("📊 View Input Data"):
        st.dataframe(input_data)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed by Team Bug Slayers | AI & ML Hackathon Project")
