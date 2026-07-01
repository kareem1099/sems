# -*- coding: utf-8 -*-
"""
Lung Cancer Risk - Streamlit Inference App (v2 - No Pollution/Radon/Occupational Exposure)
========================================================================================
Loads the saved Pipeline (StandardScaler + Logistic Regression v2).
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lung Cancer Risk Screening",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0b0f19, #1e1b4b, #0f172a);
    min-height: 100vh;
}

/* ===== Sidebar ===== */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.05);
}
[data-testid="stSidebar"] * { color: #f1f5f9 !important; }

/* ===== Headers ===== */
h1, h2, h3 { color: #ffffff !important; font-weight: 700 !important; }

/* ===== Buttons ===== */
div.stButton > button {
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 32px;
    font-size: 1.1rem;
    font-weight: 600;
    width: 100%;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(79,70,229,0.4);
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #4338ca, #6d28d9);
    box-shadow: 0 6px 30px rgba(79,70,229,0.6);
    transform: translateY(-2px);
}

/* ===== Glassmorphism cards ===== */
.glass-card {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

/* ===== Risk result ===== */
.risk-high {
    background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(185,28,28,0.1));
    border: 2px solid rgba(239,68,68,0.4);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    animation: pulse-red 2.5s infinite;
}
.risk-low {
    background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(21,128,61,0.1));
    border: 2px solid rgba(34,197,94,0.4);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    animation: pulse-green 2.5s infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 15px rgba(239,68,68,0.25); }
    50%       { box-shadow: 0 0 35px rgba(239,68,68,0.5); }
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 15px rgba(34,197,94,0.25); }
    50%       { box-shadow: 0 0 35px rgba(34,197,94,0.5); }
}

.risk-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 10px;
}
.risk-subtitle {
    font-size: 1.05rem;
    opacity: 0.9;
    margin-bottom: 18px;
}
.prob-badge {
    display: inline-block;
    background: rgba(255,255,255,0.1);
    border-radius: 50px;
    padding: 6px 20px;
    font-size: 1.3rem;
    font-weight: 700;
    color: #fff;
}

/* ===== Progress bar ===== */
.prob-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 50px;
    height: 16px;
    width: 100%;
    margin: 16px 0;
    overflow: hidden;
}
.prob-bar-fill-high {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #f97316, #ef4444);
}
.prob-bar-fill-low {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #22c55e, #10b981);
}

/* ===== Section headers ===== */
.section-header {
    color: #c7d2fe !important;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(199,210,254,0.2);
    padding-bottom: 6px;
}

hr { border-color: rgba(255,255,255,0.05) !important; }

/* Sliders & inputs */
.stSelectbox label, .stSlider label, .stNumberInput label, .stCheckbox label {
    color: #cbd5e1 !important;
    font-size: 0.9rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_lung_cancer_model_v2.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    pipeline = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Helper: build feature vector ─────────────────────────────────────────────
def build_features(inp: dict) -> pd.DataFrame:
    """Add engineered features (adjusted to remove pollution exposures)"""
    d = inp.copy()
    d["smoke_years_x_cpd"]  = d["smoking_years"]        * d["cigarettes_per_day"]
    d["pack_years_x_age"]   = d["pack_years"]            * d["age"]
    d["symptom_score"]      = (d["chronic_cough"] + d["chest_pain"] +
                                d["shortness_of_breath"] + d["fatigue"])
    d["comorbidity_score"]  = d["copd"] + d["asthma"] + d["previous_tb"]
    d["risk_exposure"]      = (d["smoker"] + d["passive_smoking"]) # Removed radon & occupational
    d["low_fev1"]           = int(d["fev1_x10"]           < 25)
    d["low_o2_sat"]         = int(d["oxygen_saturation"]  < 93)
    d["high_crp"]           = int(d["crp_level"]          > 10)
    return pd.DataFrame([d])

# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("<div style='font-size:3.8rem;margin-top:5px'>🫁</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <h1 style='margin-bottom:0;background:linear-gradient(90deg,#818cf8,#c084fc,#f472b6);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               font-size:2.4rem;font-weight:800;'>
        Lung Cancer Screening App
    </h1>
    <p style='color:#94a3b8;font-size:0.95rem;margin-top:2px;'>
        Simplified User Interface &nbsp;|&nbsp; Logistic Regression Pipeline &nbsp;|&nbsp;
        <b style='color:#34d399;'>97.9% Testing Accuracy</b>
    </p>
    """, unsafe_allow_html=True)

st.markdown("---")

if not model_loaded:
    st.error(f"**Model file v2 not found!** Run training first. Expected at: `{MODEL_PATH}`")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — INPUT FORM (Simplified - No Pollution/Radon/Occupational index)
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Patient Data")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Demographics ─────────────────────────────────────────────────────
    st.markdown("<p class='section-header'>Demographics</p>", unsafe_allow_html=True)
    age              = st.slider("Age",                18,  90, 50)
    gender           = st.selectbox("Gender",          ["Male (1)", "Female (0)"])
    gender_val       = 1 if "Male" in gender else 0

    # ── Smoking ──────────────────────────────────────────────────────────
    st.markdown("<p class='section-header'>Smoking History</p>", unsafe_allow_html=True)
    smoker           = st.checkbox("Current Smoker",        value=False)
    smoking_years    = st.slider("Smoking Years",    0,  60, 0,  disabled=not smoker)
    cigarettes_pd    = st.slider("Cigarettes / Day", 0,  60, 0,  disabled=not smoker)
    pack_years       = st.slider("Pack Years",       0,  80, 0,  disabled=not smoker)
    passive_smoking  = st.checkbox("Passive Smoking (Secondhand)", value=False)

    # ── Medical History ──────────────────────────────────────────────────
    st.markdown("<p class='section-header'>Medical History</p>", unsafe_allow_html=True)
    family_hx        = st.checkbox("Family History of Cancer", value=False)
    copd             = st.checkbox("COPD",          value=False)
    asthma           = st.checkbox("Asthma",        value=False)
    previous_tb      = st.checkbox("Previous TB",   value=False)

    # ── Symptoms ─────────────────────────────────────────────────────────
    st.markdown("<p class='section-header'>Symptoms</p>", unsafe_allow_html=True)
    chronic_cough    = st.checkbox("Chronic Cough",         value=False)
    chest_pain       = st.checkbox("Chest Pain",            value=False)
    sob              = st.checkbox("Shortness of Breath",   value=False)
    fatigue          = st.checkbox("Fatigue",               value=False)

    # ── Clinical Measurements ────────────────────────────────────────────
    st.markdown("<p class='section-header'>Clinical Measurements</p>", unsafe_allow_html=True)
    bmi              = st.slider("BMI",               10.0, 45.0, 24.0, step=0.5)
    o2_sat           = st.slider("O₂ Saturation (%)", 80,   100,  97)
    fev1             = st.slider("FEV1 × 10 (Spirometry)", 0,    50,   32)
    crp              = st.slider("CRP Level (Inflammation)", 0,    40,   3)
    xray_abnormal    = st.checkbox("Abnormal Lung X-Ray",  value=False)

    # ── Lifestyle ────────────────────────────────────────────────────────
    st.markdown("<p class='section-header'>Lifestyle</p>", unsafe_allow_html=True)
    exercise_hrs     = st.slider("Exercise Hours / Week", 0, 15, 3)
    diet_quality     = st.selectbox("Diet Quality", ["1 – Poor", "2", "3 – Average", "4", "5 – Excellent"], index=2)
    diet_val         = int(diet_quality[0])
    alcohol_units    = st.slider("Alcohol Units / Week", 0, 30, 5)
    healthcare_acc   = st.selectbox("Healthcare Access", ["1 – Low", "2", "3", "4", "5 – High"], index=2)
    healthcare_val   = int(healthcare_acc[0])

    st.markdown("<br><hr>", unsafe_allow_html=True)
    predict_btn      = st.button("🔍  Predict Risk", use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# MAIN — RESULTS
# ════════════════════════════════════════════════════════════════════════════
col_res, col_info = st.columns([3, 2], gap="large")

with col_res:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Prediction Result")

    if not predict_btn:
        st.markdown("""
        <div style='text-align:center;padding:50px 0;color:#64748b;'>
            <div style='font-size:4rem'>📊</div>
            <p style='font-size:1.15rem;margin-top:12px;'>
                Please enter the patient information in the sidebar,<br>
                then click <b style='color:#818cf8;'>Predict Risk</b> to run AI analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Build input dict (Exactly 24 original inputs without dropped features)
        inp = {
            "age":                   age,
            "gender":                gender_val,
            "smoker":                int(smoker),
            "smoking_years":         smoking_years,
            "cigarettes_per_day":    cigarettes_pd,
            "pack_years":            pack_years,
            "passive_smoking":       int(passive_smoking),
            "family_history_cancer": int(family_hx),
            "copd":                  int(copd),
            "asthma":                int(asthma),
            "previous_tb":           int(previous_tb),
            "chronic_cough":         int(chronic_cough),
            "chest_pain":            int(chest_pain),
            "shortness_of_breath":   int(sob),
            "fatigue":               int(fatigue),
            "bmi":                   bmi,
            "oxygen_saturation":     o2_sat,
            "fev1_x10":              fev1,
            "crp_level":             crp,
            "xray_abnormal":         int(xray_abnormal),
            "exercise_hours_per_week": exercise_hrs,
            "diet_quality":          diet_val,
            "alcohol_units_per_week": alcohol_units,
            "healthcare_access":     healthcare_val,
        }

        X_input  = build_features(inp)
        pred     = pipeline.predict(X_input)[0]
        proba    = pipeline.predict_proba(X_input)[0]
        risk_p   = proba[1] * 100
        no_risk_p = proba[0] * 100

        if pred == 1:
            bar_class  = "prob-bar-fill-high"
            card_class = "risk-high"
            icon       = "⚠️"
            label      = "HIGH RISK"
            color      = "#fca5a5"
            advice     = "Immediate consultation with a healthcare professional/oncologist is advised."
        else:
            bar_class  = "prob-bar-fill-low"
            card_class = "risk-low"
            icon       = "✅"
            label      = "LOW RISK"
            color      = "#6ee7b7"
            advice     = "No severe indicators detected. Standard healthy habits and regular follow-ups are recommended."

        st.markdown(f"""
        <div class='{card_class}'>
            <div class='risk-title' style='color:{color};'>{icon} {label}</div>
            <div class='risk-subtitle' style='color:#e2e8f0;'>{advice}</div>
            <div class='prob-badge'>Probability score: {risk_p:.1f}%</div>
            <div class='prob-bar-bg'>
                <div class='{bar_class}' style='width:{risk_p:.1f}%'></div>
            </div>
            <p style='color:#94a3b8;font-size:0.85rem;margin-top:6px;'>
                Negative Probability: {no_risk_p:.1f}% &nbsp;|&nbsp; Positive Probability: {risk_p:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Score",  f"{risk_p:.1f}%")
        m2.metric("Testing Accuracy", "97.9%")
        m3.metric("Model ROC-AUC", "0.9978")

    st.markdown("</div>", unsafe_allow_html=True)

with col_info:
    st.markdown("""
    <div class='glass-card'>
        <h3 style='color:#818cf8;margin-top:0;'>⚙️ Model Specifications</h3>
        <table style='width:100%;color:#cbd5e1;font-size:0.88rem;border-collapse:collapse;'>
            <tr><td style='padding:6px 0;color:#64748b;'>Model Type</td>
                <td style='font-weight:600;color:#e2e8f0;'>Logistic Regression (Pipeline)</td></tr>
            <tr><td style='padding:6px 0;color:#64748b;'>Scaling</td>
                <td style='font-weight:600;color:#e2e8f0;'>StandardScaler (Embedded)</td></tr>
            <tr><td style='padding:6px 0;color:#64748b;'>Original Features</td>
                <td style='font-weight:600;color:#e2e8f0;'>24 (Dropped 5 inputs)</td></tr>
            <tr><td style='padding:6px 0;color:#64748b;'>Derived Features</td>
                <td style='font-weight:600;color:#e2e8f0;'>8 engineered features</td></tr>
            <tr><td style='padding:6px 0;color:#64748b;'>Testing Accuracy</td>
                <td style='font-weight:600;color:#34d399;'>97.90%</td></tr>
            <tr><td style='padding:6px 0;color:#64748b;'>ROC-AUC</td>
                <td style='font-weight:600;color:#34d399;'>0.9978</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-card'>
        <h3 style='color:#c084fc;margin-top:0;'>💡 User Input Convenience</h3>
        <p style='font-size:0.88rem;color:#94a3b8;line-height:1.6;'>
            This version does not require <b>Radon Exposure</b>, <b>Air Pollution Index</b>, <b>Occupational Exposure</b>, <b>Education Years</b>, or <b>Income Level</b>. This simplifies inputs for patients.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);
                border-radius:12px;padding:14px;font-size:0.8rem;color:#fef08a;'>
        ⚠️ <b>Clinical Disclaimer:</b> AI prediction models are diagnostic supports, not conclusive clinical confirmations.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<p style='text-align:center;color:#475569;font-size:0.8rem;'>
    Lung Cancer Risk Predictor App v2 &nbsp;|&nbsp; Developed on Cleaned Subset &nbsp;|&nbsp; 2026
</p>
""", unsafe_allow_html=True)
