"""
═══════════════════════════════════════════════════════════════════
  💓  AI Cardiac Risk Prediction System
═══════════════════════════════════════════════════════════════════
"""

import os, json, pickle
import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# ─── Load ML Model ───────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
SEGMENT_LEN = 181

# LSTM Model Definition (must match training)
import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True,
                           bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

ML_AVAILABLE = False
ml_model = None
scaler = None
label_classes = None
model_type = "None"

try:
    # Try loading LSTM model first (client requested)
    lstm_path = os.path.join(MODEL_DIR, "cardiac_lstm_model.pth")
    if os.path.exists(lstm_path):
        device = torch.device('cpu')
        ml_model = BiLSTMClassifier().to(device)
        ml_model.load_state_dict(torch.load(lstm_path, map_location=device))
        ml_model.eval()
        model_type = "BiLSTM"
        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "label_classes.pkl"), "rb") as f:
            label_classes = pickle.load(f)
        ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Cardiac Risk AI",
    layout="wide",
    page_icon="💓",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════
#  MODERN MEDICAL UI STYLING
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #f8fafc;
}

/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Modern Header */
.modern-header {
    background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 50%, #14b8a6 100%);
    padding: 2.5rem 3rem;
    border-radius: 20px;
    margin-bottom: 2.5rem;
    box-shadow: 0 10px 40px rgba(14, 165, 233, 0.25);
    position: relative;
    overflow: hidden;
}

.modern-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: rotation 20s linear infinite;
}

@keyframes rotation {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.header-content {
    position: relative;
    z-index: 1;
}

.app-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 42px;
    font-weight: 700;
    color: white;
    margin: 0;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 16px;
    justify-content: center;
}

.app-subtitle {
    font-size: 16px;
    color: rgba(255,255,255,0.9);
    margin-top: 8px;
    text-align: center;
    font-weight: 500;
}

/* Modern Card */
.modern-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.modern-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(14, 165, 233, 0.15);
    border-color: rgba(14, 165, 233, 0.3);
}

.card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.1);
}

.card-icon {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background: linear-gradient(135deg, #0ea5e9, #06b6d4);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    color: #f8fafc;
    margin: 0;
    font-family: 'Space Grotesk', sans-serif;
}

/* Status Badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    margin: 6px 6px 6px 0;
    letter-spacing: 0.02em;
}

.badge-success {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-danger {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.badge-info {
    background: rgba(14, 165, 233, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(14, 165, 233, 0.3);
}

.badge-warning {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* Form Controls */
.stNumberInput input, .stSelectbox select, .stFileUploader {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1.5px solid rgba(148, 163, 184, 0.3) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
    padding: 12px 16px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stNumberInput input:focus, .stSelectbox select:focus {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1) !important;
}

.stCheckbox {
    padding: 10px 0;
}

.stCheckbox label {
    font-weight: 600 !important;
    color: #e2e8f0 !important;
    font-size: 15px !important;
}

.stCheckbox input:checked ~ label {
    color: #0ea5e9 !important;
}

/* Primary Button */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #06b6d4) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 14px 28px !important;
    border: none !important;
    border-radius: 10px !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(14, 165, 233, 0.3) !important;
    letter-spacing: 0.02em !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(14, 165, 233, 0.4) !important;
}

/* Alert Messages */
.alert-message {
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    margin: 1.25rem 0;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 15px;
}

.alert-success {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #6ee7b7;
}

.alert-danger {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #fca5a5;
}

.alert-warning {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fcd34d;
}

.alert-info {
    background: rgba(14, 165, 233, 0.1);
    border: 1px solid rgba(14, 165, 233, 0.3);
    color: #7dd3fc;
}

/* File Uploader */
.stFileUploader {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.5), rgba(51, 65, 85, 0.5)) !important;
    border: 2px dashed rgba(148, 163, 184, 0.3) !important;
    border-radius: 12px !important;
    padding: 2rem !important;
    transition: all 0.3s ease !important;
}

.stFileUploader:hover {
    border-color: #0ea5e9 !important;
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.05), rgba(6, 182, 212, 0.05)) !important;
}

/* Metric Grid */
.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.info-item {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
}

.info-value {
    font-size: 28px;
    font-weight: 700;
    color: #0ea5e9;
    font-family: 'Space Grotesk', sans-serif;
}

.info-label {
    font-size: 13px;
    color: #94a3b8;
    font-weight: 600;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Divider */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.3), transparent);
    margin: 2rem 0;
}

/* Image Preview */
.stImage {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
}
</style>
""", unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════
#  ML ECG ANALYSIS
# ═══════════════════════════════════════════════════════════════════
def ml_analyze_ecg_image(img):
    gray = img.convert("L")
    arr  = np.array(gray, dtype=np.float32)
    h, w = arr.shape
    band = arr[int(h * 0.2):int(h * 0.8), :]
    waveform = np.argmin(band, axis=0).astype(np.float32)
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
    
    num_segs = len(waveform) // SEGMENT_LEN
    if num_segs == 0:
        padded = np.zeros(SEGMENT_LEN, dtype=np.float32)
        padded[:min(len(waveform), SEGMENT_LEN)] = waveform[:SEGMENT_LEN]
        segments = [padded]
    else:
        segments = [waveform[i*SEGMENT_LEN:(i+1)*SEGMENT_LEN]
                    for i in range(num_segs)]
    
    segments_array = np.array(segments)
    
    
    # PyTorch LSTM model - uses raw segments (181 features)
    segments_scaled = scaler.transform(segments_array)
    
    device = torch.device('cpu')
    X_tensor = torch.FloatTensor(segments_scaled).to(device)
    with torch.no_grad():
        outputs = ml_model(X_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    
    avg_prob = probs.mean(axis=0)
    pred_cls = int(np.argmax(avg_prob))
    confidence = float(avg_prob[pred_cls]) * 100
    
    return label_classes[pred_cls], round(confidence, 1), avg_prob


# ═══════════════════════════════════════════════════════════════════
#  RISK CALCULATION
# ═══════════════════════════════════════════════════════════════════
def calculate_risk(data):
    score = 0
    if data.get("ml_class") == "Abnormal": score += 3
    if data["hr"] < 50 or data["hr"] > 100: score += 2
    if data["pain"]: score += 2
    if data["breath"]: score += 2
    if data["dizzy"]: score += 1
    if data["drug"]: score += 1
    if data["dose"] == "High": score += 2
    if data["age"] > 60: score += 2
    if data["history"]: score += 3
    return min(int((score / 16) * 100), 100)

def predict_diseases(data, risk):
    diseases = []
    if data.get("ml_class") == "Abnormal":
        diseases.append("Arrhythmia")
    if data["hr"] > 100:
        diseases.append("Tachycardia")
    if data["hr"] < 50:
        diseases.append("Bradycardia")
    if data["pain"] and data["breath"]:
        diseases.append("Coronary Artery Disease (CAD)")
    if risk >= 70 and (data["breath"] or data["history"]):
        diseases.append("Heart Failure Risk")
    if not diseases:
        diseases.append("No significant cardiac abnormality detected")
    return diseases

# ═══════════════════════════════════════════════════════════════════
#  SPEEDOMETER GAUGE
# ═══════════════════════════════════════════════════════════════════
def show_speedometer(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 56, "color": "white", "family": "Plus Jakarta Sans"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#667eea", "tickfont": {"color": "white"}},
            "bar": {"color": "#667eea", "thickness": 0.8},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 2,
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, 40],  "color": "rgba(16, 185, 129, 0.3)"},
                {"range": [40, 70], "color": "rgba(245, 158, 11, 0.3)"},
                {"range": [70, 100],"color": "rgba(239, 68, 68, 0.3)"}
            ],
        }
    ))
    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "family": "Plus Jakarta Sans"}
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class='modern-header'>
    <div class='header-content'>
        <h1 class='app-title'><span>💓</span> Cardiac Risk AI</h1>
        <p class='app-subtitle'>Advanced ML-powered ECG Analysis · Real-time Risk Assessment · LSTM Deep Learning</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── ECG Upload Section ───
st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
st.markdown("""
<div class='card-header'>
    <div class='card-icon'>📤</div>
    <h2 class='card-title'>Upload ECG Report</h2>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your ECG image here (JPG or PNG)",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed"
)

ml_class = None
ml_conf = 0

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, use_column_width=True)
    
    if ML_AVAILABLE and st.checkbox("✅ This is a valid ECG report", value=False):
        with st.spinner("🧠 AI analyzing ECG waveform..."):
            ml_class, ml_conf, probs = ml_analyze_ecg_image(img)
        
        col1, col2 = st.columns(2)
        with col1:
            if ml_class == "Normal":
                st.markdown(f"<span class='status-badge badge-success'>✓ {ml_class}</span>", 
                           unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='status-badge badge-danger'>⚠ {ml_class}</span>", 
                           unsafe_allow_html=True)
        with col2:
            st.markdown(f"<span class='status-badge badge-info'>🎯 {ml_conf}% Confidence</span>", 
                       unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ─── Patient Form ───
st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
st.markdown("""
<div class='card-header'>
    <div class='card-icon'>🩺</div>
    <h2 class='card-title'>Patient Information</h2>
</div>
""", unsafe_allow_html=True)

with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 0, 130, 45)
        heart_rate = st.number_input("Heart Rate (BPM)", 20, 250, 72)
    
    with col2:
        chest_pain = st.checkbox("Chest Pain")
        shortness_breath = st.checkbox("Shortness of Breath")
    
    with col3:
        dizziness = st.checkbox("Dizziness")
        history_hd = st.checkbox("History of Heart Disease")
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    col4, col5 = st.columns(2)
    with col4:
        taking_drug = st.checkbox("Currently Taking Medication")
    with col5:
        dosage = st.selectbox("Dosage Level", ["Low", "Medium", "High"])
    
    submitted = st.form_submit_button("🔮 Analyze Cardiac Risk")

st.markdown("</div>", unsafe_allow_html=True)

# ─── Results ───
if submitted:
    patient_data = {
        "ml_class": ml_class,
        "hr": heart_rate,
        "pain": chest_pain,
        "breath": shortness_breath,
        "dizzy": dizziness,
        "drug": taking_drug,
        "dose": dosage,
        "age": age,
        "history": history_hd,
    }
    
    risk = calculate_risk(patient_data)
    diseases = predict_diseases(patient_data, risk)
    
    # Risk Gauge
    st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card-header'>
        <div class='card-icon'>📊</div>
        <h2 class='card-title'>Cardiac Risk Score</h2>
    </div>
    """, unsafe_allow_html=True)
    show_speedometer(risk)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Predictions
    st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card-header'>
        <div class='card-icon'>🔍</div>
        <h2 class='card-title'>Detected Conditions</h2>
    </div>
    """, unsafe_allow_html=True)
    
    for disease in diseases:
        if "No significant" in disease:
            st.markdown(f"<div class='alert-message alert-success'>✅ {disease}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='alert-message alert-warning'>⚠️ {disease}</div>", unsafe_allow_html=True)
    
    if risk >= 70:
        st.markdown("<div class='alert-message alert-danger'>🚨 HIGH RISK — Immediate medical consultation recommended</div>", unsafe_allow_html=True)
    elif risk >= 40:
        st.markdown("<div class='alert-message alert-warning'>⚠️ MODERATE RISK — Regular monitoring advised</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='alert-message alert-success'>✅ LOW RISK — No immediate concerns detected</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
