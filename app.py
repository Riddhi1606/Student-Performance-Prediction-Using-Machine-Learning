import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.big-title {
    font-size: 2.5rem; font-weight: 800;
    color: #1f77b4; text-align: center;
    padding: 1rem 0 0.2rem 0;
}
.subtitle {
    font-size: 1.05rem; color: #666;
    text-align: center; margin-bottom: 1.5rem;
}
.metric-box {
    padding: 1.5rem; border-radius: 15px;
    text-align: center; color: white;
}
.metric-value { font-size: 2.8rem; font-weight: 800; }
.metric-label { font-size: 0.95rem; opacity: 0.9; margin-top: 4px; }
.section-header {
    font-size: 1.3rem; font-weight: 700;
    color: #333; margin: 1.2rem 0 0.5rem 0;
    border-left: 4px solid #1f77b4;
    padding-left: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL & SCALER ----------------
model  = joblib.load("xgboost_student_model.pkl")
scaler = joblib.load("scaler.pkl")

def minmax_scale(study_hours, attendance, sleep):
    scaled = scaler.transform([[study_hours, attendance, sleep]])
    return scaled[0][0], scaled[0][1], scaled[0][2]

# ---------------- HEADER ----------------
st.markdown('<div class="big-title">🎓 Student Performance Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">This application predicts student final marks using an XGBoost machine learning model.<br>'
    'Fill in the student details below and click <b>Predict Final Marks</b>.</div>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 About This Project")
st.sidebar.info("""
**ML Model:** XGBoost Regressor

**Task:** Regression — Final Marks Prediction

**Dataset:** Student Performance Dataset

**Tech Stack:** Python · Scikit-learn · XGBoost · Streamlit · Plotly
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.success("R² Score: **0.97**")
st.sidebar.info("MAE: **~2.1**")
st.sidebar.markdown("---")
st.sidebar.markdown("**👩‍💻 Developed by Team Bug Slayers 🐛**")
st.sidebar.caption("AI & Data Science Project | Arya College of Engineering & IT")

# ---------------- INPUT SECTION ----------------
st.markdown('<div class="section-header">📝 Enter Student Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Personal Info**")
    gender_label = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender_label == "Male" else 0

    school_label = st.selectbox("School Type", ["Public", "Private"])
    school_type = 1 if school_label == "Public" else 0

    parent_label = st.selectbox("Parent Education", ["None", "Graduate (Grad)", "Post Graduate (PG)"])
    parent_map = {"None": 0, "Graduate (Grad)": 1, "Post Graduate (PG)": 2}
    parent_education = parent_map[parent_label]

    extra_label = st.selectbox("Extracurricular Activities", ["No", "Yes"])
    extracurricular = 0 if extra_label == "No" else 1

with col2:
    st.markdown("**📚 Academic Info**")
    previous_score  = st.slider("Previous Score", 20, 95, 60)
    study_hours_raw = st.slider("Study Hours Daily", 1, 11, 5)
    attendance_raw  = st.slider("Attendance %", 40, 100, 75)
    sleep_hours_raw = st.slider("Sleep Hours", 4, 10, 7)

with col3:
    st.markdown("**📊 Live Input Summary**")
    st.metric("Study Hours",     f"{study_hours_raw} hrs/day")
    st.metric("Attendance",      f"{attendance_raw}%")
    st.metric("Sleep Hours",     f"{sleep_hours_raw} hrs/night")
    st.metric("Previous Score",  f"{previous_score}/100")
    st.metric("Extracurricular", extra_label)

# ---------------- RADAR CHART ----------------
st.markdown("---")
st.markdown('<div class="section-header">📊 Student Profile Overview</div>', unsafe_allow_html=True)

radar_names  = ["Study Hours", "Attendance", "Sleep", "Prev Score", "Extracurricular", "Parent Edu"]
radar_values = [
    study_hours_raw / 11,
    attendance_raw  / 100,
    sleep_hours_raw / 10,
    previous_score  / 95,
    extracurricular,
    parent_education / 2
]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=radar_values + [radar_values[0]],
    theta=radar_names + [radar_names[0]],
    fill='toself',
    fillcolor='rgba(31, 119, 180, 0.2)',
    line=dict(color='#1f77b4', width=2),
    name='Student Profile'
))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=False,
    title=dict(text="Student Profile Radar Chart", x=0.5),
    height=320,
    margin=dict(t=50, b=10)
)
st.plotly_chart(fig_radar, use_container_width=True)

# ---------------- PREDICTION BUTTON ----------------
st.markdown("---")
predict_btn = st.button("🚀 Predict Final Marks", use_container_width=True, type="primary")

if predict_btn:

    # Step 1 — Scale numeric features using saved scaler
    study_scaled, attendance_scaled, sleep_scaled = minmax_scale(
        study_hours_raw, attendance_raw, sleep_hours_raw
    )

    # Step 2 — Feature engineering on SCALED values (matches notebook exactly)
    study_effectiveness = study_scaled * attendance_scaled
    sleep_study_ratio   = sleep_scaled / max(study_scaled, 0.001)
    engagement_score    = study_scaled * 0.5 + attendance_scaled * 0.3 + extracurricular * 0.2

    # Step 3 — Build input dataframe in exact column order from notebook
    input_data = pd.DataFrame({
        "gender":              [gender],
        "school_type":         [school_type],
        "study_hours_daily":   [study_scaled],
        "attendance_pct":      [attendance_scaled],
        "sleep_hours":         [sleep_scaled],
        "previous_score":      [previous_score],
        "extracurricular":     [extracurricular],
        "parent_education":    [parent_education],
        "study_effectiveness": [study_effectiveness],
        "sleep_study_ratio":   [sleep_study_ratio],
        "engagement_score":    [engagement_score]
    })

    with st.spinner("🤖 Analysing student data..."):
        prediction = float(model.predict(input_data)[0])
        prediction = max(0, min(100, prediction))

    prediction_int = int(round(prediction))

    # ---------------- PREDICTION INDEX ----------------
    # Weighted score of all input factors normalized to 0-100
    prediction_index = int(round(
        (study_hours_raw / 11) * 25 +
        (attendance_raw  / 100) * 25 +
        (previous_score  / 95) * 30 +
        (sleep_hours_raw / 10) * 10 +
        extracurricular        * 5  +
        (parent_education / 2) * 5
    ))
    prediction_index = max(0, min(100, prediction_index))

    # ---------------- RESULTS ----------------
    st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)

    # Grade & color
    if prediction_int >= 90:
        grade, grade_color = "A+", "linear-gradient(135deg,#11998e,#38ef7d)"
    elif prediction_int >= 80:
        grade, grade_color = "A",  "linear-gradient(135deg,#56ab2f,#a8e063)"
    elif prediction_int >= 70:
        grade, grade_color = "B",  "linear-gradient(135deg,#1f77b4,#4facfe)"
    elif prediction_int >= 60:
        grade, grade_color = "C",  "linear-gradient(135deg,#f7971e,#ffd200)"
    elif prediction_int >= 50:
        grade, grade_color = "D",  "linear-gradient(135deg,#f37335,#fda085)"
    else:
        grade, grade_color = "F",  "linear-gradient(135deg,#cb2d3e,#ef473a)"

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg,#667eea,#764ba2);">
            <div class="metric-value">{prediction_int}/100</div>
            <div class="metric-label">🎯 Predicted Final Score</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg,#f093fb,#f5576c);">
            <div class="metric-value">{prediction_index}/100</div>
            <div class="metric-label">📊 Prediction Index</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="metric-box" style="background: {grade_color};">
            <div class="metric-value">{grade}</div>
            <div class="metric-label">🏅 Grade</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(prediction_int)

    if prediction_int >= 90:
        st.success("🏆 **Excellent Performance** — Outstanding result! Keep it up!")
    elif prediction_int >= 75:
        st.info("🌟 **Good Performance** — Strong result! A little more effort for excellence.")
    elif prediction_int >= 50:
        st.warning("📚 **Average Performance** — There is room for improvement. Focus on weak areas.")
    else:
        st.error("⚠️ **Needs Improvement** — Please seek additional support and increase study time.")

    # ---------------- SCORE GAUGE ----------------
    st.markdown('<div class="section-header">📉 Score Gauge</div>', unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction_int,
        delta={'reference': 70, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#667eea", 'thickness': 0.3},
            'steps': [
                {'range': [0,  50], 'color': "#ffcccc"},
                {'range': [50, 75], 'color': "#fff3cd"},
                {'range': [75, 90], 'color': "#cce5ff"},
                {'range': [90,100], 'color': "#d4edda"},
            ],
            'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.75, 'value': 50}
        },
        title={'text': f"Predicted Score: {prediction_int}/100", 'font': {'size': 16}}
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=50, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ---------------- FEATURE IMPORTANCE ----------------
    st.markdown('<div class="section-header">📈 Feature Importance</div>', unsafe_allow_html=True)
    feat_names = [
        "Gender", "School Type", "Study Hours", "Attendance %",
        "Sleep Hours", "Previous Score", "Extracurricular",
        "Parent Education", "Study Effectiveness", "Sleep/Study Ratio", "Engagement Score"
    ]
    feat_df = pd.DataFrame({
        "Feature":    feat_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig_imp = px.bar(
        feat_df, x="Importance", y="Feature", orientation='h',
        color="Importance", color_continuous_scale="Blues",
        title="Which factors influence the prediction most?",
        labels={"Importance": "Importance Score", "Feature": ""}
    )
    fig_imp.update_layout(height=380, margin=dict(t=50, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    # ---------------- AI EXPLANATION ----------------
    st.markdown('<div class="section-header">🧠 AI Explanation — Why this prediction?</div>', unsafe_allow_html=True)

    explanations = []

    if study_hours_raw >= 7:
        explanations.append(("✅", "**Study Hours are high** — Consistent study time is the biggest driver of academic success."))
    elif study_hours_raw >= 4:
        explanations.append(("⚠️", "**Study Hours are moderate** — Increasing daily study time could significantly improve the score."))
    else:
        explanations.append(("❌", "**Study Hours are very low** — This is the primary reason for a low prediction. Study more daily."))

    if attendance_raw >= 85:
        explanations.append(("✅", "**Excellent attendance** — High attendance correlates strongly with better understanding and marks."))
    elif attendance_raw >= 70:
        explanations.append(("⚠️", "**Attendance is average** — Try not to miss classes; each session builds on the previous one."))
    else:
        explanations.append(("❌", "**Low attendance** — Missing classes is negatively impacting this prediction significantly."))

    if previous_score >= 70:
        explanations.append(("✅", "**Strong previous score** — Past performance shows consistent academic ability."))
    elif previous_score >= 50:
        explanations.append(("⚠️", "**Average previous score** — Indicates potential but suggests inconsistency that needs to be addressed."))
    else:
        explanations.append(("❌", "**Low previous score** — This is weighing down the prediction. Targeted revision of past topics is essential."))

    if sleep_hours_raw >= 7:
        explanations.append(("✅", "**Adequate sleep** — Good sleep supports memory consolidation and focus during study."))
    elif sleep_hours_raw >= 6:
        explanations.append(("⚠️", "**Sleep is slightly low** — Aim for at least 7–8 hours for optimal brain performance."))
    else:
        explanations.append(("❌", "**Insufficient sleep** — Poor sleep is affecting concentration and retention ability."))

    if extracurricular == 1:
        explanations.append(("✅", "**Extracurricular participation** — Shows a well-rounded student with good time management."))
    else:
        explanations.append(("ℹ️", "**No extracurricular** — Joining activities can improve engagement and soft skills."))

    if parent_education >= 2:
        explanations.append(("✅", "**Highly educated parents** — Supportive academic home environment is a positive influence."))
    elif parent_education == 1:
        explanations.append(("ℹ️", "**Moderate parent education** — Moderate home academic support available."))
    else:
        explanations.append(("ℹ️", "**Limited parent education** — Student is self-driven; access to additional resources is recommended."))

    for icon, text in explanations:
        st.markdown(f"{icon} {text}")

    # ---------------- INPUT SUMMARY ----------------
    with st.expander("📋 View Processed Input Data"):
        display_df = pd.DataFrame({
            "Feature": [
                "Gender", "School Type", "Study Hours/day", "Attendance %",
                "Sleep Hours", "Previous Score", "Extracurricular",
                "Parent Education", "Study Effectiveness*",
                "Sleep/Study Ratio*", "Engagement Score*"
            ],
            "Raw Value": [
                gender_label, school_label, f"{study_hours_raw} hrs",
                f"{attendance_raw}%", f"{sleep_hours_raw} hrs",
                previous_score, extra_label, parent_label,
                f"{study_effectiveness:.4f}",
                f"{sleep_study_ratio:.4f}",
                f"{engagement_score:.4f}"
            ]
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption("* Engineered features computed from scaled values — matches exact notebook pipeline.")
        st.info(
    "Predictions are generated using an XGBoost regression model trained on student performance data."
)

# ---------------- FOOTER ----------------
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("🐛 **Team Bug Slayers**")
with col_f2:
    st.caption("🎓 Arya College of Engineering & IT")
with col_f3:
    st.caption("🤖 AI & Data Science Project | 2025")
