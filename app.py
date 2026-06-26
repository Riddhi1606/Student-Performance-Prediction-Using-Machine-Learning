import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.big-title { font-size: 2.5rem; font-weight: 800; color: #1f77b4; text-align: center; }
.subtitle  { font-size: 1.1rem; color: #555; text-align: center; margin-bottom: 2rem; }
.metric-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem; border-radius: 15px; text-align: center; color: white;
}
.metric-value { font-size: 3rem; font-weight: 800; }
.metric-label { font-size: 1rem; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("xgboost_student_model.pkl")

# ---------------- HEADER ----------------
st.markdown('<div class="big-title">🎓 Student Performance Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">This application predicts student final marks using an XGBoost machine learning model.<br>Fill in the details below and click <b>Predict</b>.</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 About Project")
st.sidebar.info("""
- ML Model: XGBoost Regressor  
- Task: Regression (Final Marks Prediction)  
- Dataset: Student Performance Dataset  
- Built using: Streamlit + Python  
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Team Bug Slayers** 🐛")
st.sidebar.markdown("AI & ML Project | 2025")

# ---------------- INPUT UI ----------------
st.markdown("### 📝 Enter Student Details")
col1, col2, col3 = st.columns(3)

with col1:
    gender_label = st.selectbox("👤 Gender", ["Male", "Female"])
    gender = 0 if gender_label == "Male" else 1

    school_label = st.selectbox("🏫 School Type", ["Public", "Private"])
    school_type = 0 if school_label == "Public" else 1

    study_hours_daily = st.slider("📖 Study Hours Daily", 0, 12, 5)
    attendance_pct = st.slider("📅 Attendance %", 0, 100, 75) / 100

with col2:
    sleep_hours = st.slider("😴 Sleep Hours", 0, 12, 7)
    previous_score = st.slider("📊 Previous Score", 0, 100, 60)

    extra_label = st.selectbox("🏅 Extracurricular", ["No", "Yes"])
    extracurricular = 0 if extra_label == "No" else 1

    parent_label = st.selectbox("👨‍👩‍👧 Parent Education", ["High School", "Graduate", "Post Graduate"])
    parent_map = {"High School": 0, "Graduate": 1, "Post Graduate": 2}
    parent_education = parent_map[parent_label]

with col3:
    study_effectiveness = st.slider("⚡ Study Effectiveness (1-10)", 0, 10, 6) / 10
    sleep_study_ratio = st.slider("⚖️ Sleep/Study Ratio (1-50)", 0, 50, 15) / 10
    engagement_score = st.slider("🎯 Engagement Score (1-10)", 0, 10, 6) / 10

# ---------------- REAL-TIME GAUGE ----------------
st.markdown("---")
st.markdown("### 📊 Real-Time Input Overview")

feature_names = [
    "Study Hours", "Attendance %", "Sleep Hours",
    "Previous Score", "Study Effectiveness", "Sleep/Study Ratio", "Engagement Score"
]
feature_values = [
    study_hours_daily / 12,
    attendance_pct,
    sleep_hours / 12,
    previous_score / 100,
    study_effectiveness,
    min(sleep_study_ratio / 5, 1.0),
    engagement_score
]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=feature_values + [feature_values[0]],
    theta=feature_names + [feature_names[0]],
    fill='toself',
    fillcolor='rgba(102, 126, 234, 0.3)',
    line=dict(color='#667eea', width=2),
    name='Student Profile'
))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=False,
    title="Student Profile Radar Chart",
    height=350,
    margin=dict(t=60, b=20)
)
st.plotly_chart(fig_radar, use_container_width=True)

# ---------------- PREDICTION ----------------
st.markdown("---")
predict_btn = st.button("🚀 Predict Final Marks", use_container_width=True)

if predict_btn:
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

    with st.spinner("🤖 Analysing student data..."):
        prediction = float(model.predict(input_data)[0])
        prediction = max(0, min(100, prediction))

    # Round to whole number
    prediction_int = int(round(prediction))

    # ---------------- RESULT BOX ----------------
    st.markdown("---")
    st.markdown("### 🎯 Prediction Results")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{prediction_int}</div>
            <div class="metric-label">Predicted Score / 100</div>
        </div>""", unsafe_allow_html=True)

    # Confidence score
    confidence = int(50 + abs(prediction - 50))
    confidence = min(confidence, 99)
    with r2:
        st.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-value">{confidence}%</div>
            <div class="metric-label">Confidence Score</div>
        </div>""", unsafe_allow_html=True)

    # Grade
    if prediction_int >= 90:
        grade, color = "A+", "#2ecc71"
    elif prediction_int >= 80:
        grade, color = "A", "#27ae60"
    elif prediction_int >= 70:
        grade, color = "B", "#3498db"
    elif prediction_int >= 60:
        grade, color = "C", "#f39c12"
    elif prediction_int >= 50:
        grade, color = "D", "#e67e22"
    else:
        grade, color = "F", "#e74c3c"

    with r3:
        st.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-value">{grade}</div>
            <div class="metric-label">Grade</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Progress bar
    st.progress(prediction_int)

    # Performance category
    if prediction_int >= 90:
        st.success("🏆 Performance: Excellent — Outstanding result!")
    elif prediction_int >= 75:
        st.info("🌟 Performance: Good — Keep it up!")
    elif prediction_int >= 50:
        st.warning("📚 Performance: Average — Room for improvement!")
    else:
        st.error("⚠️ Performance: Needs Improvement — Focus on weak areas!")

    # ---------------- SCORE GAUGE ----------------
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction_int,
        delta={'reference': 70, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 50],  'color': "#ffcccc"},
                {'range': [50, 75], 'color': "#fff3cd"},
                {'range': [75, 90], 'color': "#cce5ff"},
                {'range': [90, 100],'color': "#d4edda"},
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
        },
        title={'text': "Final Score Gauge"}
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=60, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ---------------- FEATURE IMPORTANCE ----------------
    st.markdown("### 📈 Feature Importance Chart")

    feat_names = [
        "Gender", "School Type", "Study Hours", "Attendance %",
        "Sleep Hours", "Previous Score", "Extracurricular",
        "Parent Education", "Study Effectiveness", "Sleep/Study Ratio", "Engagement Score"
    ]

    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": feat_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig_imp = px.bar(
        feat_df, x="Importance", y="Feature",
        orientation='h',
        color="Importance",
        color_continuous_scale="viridis",
        title="What factors matter most for prediction?",
        labels={"Importance": "Importance Score"}
    )
    fig_imp.update_layout(height=400, margin=dict(t=60, b=20))
    st.plotly_chart(fig_imp, use_container_width=True)

    # ---------------- AI EXPLANATION ----------------
    st.markdown("### 🧠 AI Explanation — Why is this prediction high/low?")

    explanations = []

    if study_hours_daily >= 6:
        explanations.append("✅ **High study hours** are positively contributing to the score.")
    else:
        explanations.append("⚠️ **Low study hours** are pulling the score down. Try studying more.")

    if attendance_pct >= 0.75:
        explanations.append("✅ **Good attendance** is boosting the predicted score.")
    else:
        explanations.append("⚠️ **Poor attendance** is negatively affecting the result.")

    if previous_score >= 70:
        explanations.append("✅ **Strong previous score** indicates consistent performance.")
    else:
        explanations.append("⚠️ **Low previous score** is a key reason for lower prediction.")

    if study_effectiveness >= 0.7:
        explanations.append("✅ **High study effectiveness** shows quality learning habits.")
    else:
        explanations.append("⚠️ **Low study effectiveness** — focus on quality, not just quantity.")

    if engagement_score >= 0.7:
        explanations.append("✅ **High engagement score** reflects strong classroom involvement.")
    else:
        explanations.append("⚠️ **Low engagement** — try participating more actively in class.")

    if extracurricular == 1:
        explanations.append("✅ **Extracurricular participation** shows well-rounded development.")

    if sleep_hours < 6:
        explanations.append("⚠️ **Insufficient sleep** may be affecting focus and retention.")
    elif sleep_hours >= 7:
        explanations.append("✅ **Adequate sleep** supports better cognitive performance.")

    for exp in explanations:
        st.markdown(f"- {exp}")

    # ---------------- INPUT DATA ----------------
    with st.expander("📊 View Raw Input Data"):
        st.dataframe(input_data)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed by Team Bug Slayers 🐛 | AI & ML Project | Arya College of Engineering & IT")
