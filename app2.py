import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import plotly.graph_objects as go

# ================= PAGE CONFIG =================
st.set_page_config(page_title="AI Cardiac Risk System", layout="centered")

# ================= CSS =================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #1e293b, #020617);
    color: #e5e7eb;
}
@keyframes heartbeat {
    0% {transform:scale(1);opacity:.2;}
    25% {transform:scale(1.15);opacity:.45;}
    50% {transform:scale(1);opacity:.2;}
}
.heart-bg{
    position:fixed;
    top:55%;
    left:50%;
    transform:translate(-50%,-50%);
    font-size:260px;
    color:rgba(239,68,68,.18);
    animation:heartbeat 1.4s infinite;
    z-index:0;
}
.app-header{
    background:linear-gradient(90deg,#ec4899,#ef4444,#f97316);
    padding:26px;
    border-radius:0 0 24px 24px;
    box-shadow:0 18px 45px rgba(0,0,0,.6);
}
.app-title{
    font-size:42px;
    font-weight:900;
    color:white;
}
section[data-testid="stForm"]{
    background:rgba(15,23,42,.85);
    border-radius:24px;
    padding:28px;
    box-shadow:0 25px 60px rgba(0,0,0,.7);
}
.stButton>button{
    width:100%;
    background:#22c55e;
    color:#052e16;
    font-weight:800;
    padding:14px;
    border-radius:16px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='heart-bg'>❤️</div>", unsafe_allow_html=True)
st.markdown("""
<div class="app-header">
  <div class="app-title">💓 AI Cardiac Risk Prediction System</div>
  <div>ECG Image Analysis + Future Heart Disease Risk</div>
</div>
""", unsafe_allow_html=True)

# ================= ECG ANALYSIS (BASELINE) =================
def analyze_ecg_image(img):
    gray = img.convert("L")
    arr = np.array(gray)

    # Focus only on the waveform area (middle band)
    h, w = arr.shape
    band = arr[int(h*0.35):int(h*0.65), :]

    # Smoothness check
    diff = np.abs(np.diff(band.astype(float), axis=1))
    mean_diff = np.mean(diff)

    # Heuristic decision
    if mean_diff < 12:
        # smoother signal → normal rhythm
        confidence = int(75 + (12 - mean_diff) * 2)
        return "Normal", min(confidence, 95)
    else:
        # irregular spacing / noisy rhythm
        confidence = int(70 + (mean_diff - 12) * 1.5)
        return "Irregular", min(confidence, 95)

# ================= RISK SCORE =================
def calculate_risk(data):
    score = 0
    if data["ecg"] == "Irregular": score += 3
    if data["hr"] < 50 or data["hr"] > 100: score += 2
    if data["pain"]: score += 2
    if data["breath"]: score += 2
    if data["dizzy"]: score += 1
    if data["drug"]: score += 1
    if data["dose"] == "High": score += 2
    if data["age"] > 60: score += 2
    if data["history"]: score += 3
    return min(int((score / 14) * 100), 100)

# ================= HEART DISEASE PREDICTION =================
def predict_diseases(data, risk):
    diseases = []

    if data["ecg"] == "Irregular":
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

# ================= SPEEDOMETER =================
def show_speedometer(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 40], "color": "#22c55e"},
                {"range": [40, 70], "color": "#f59e0b"},
                {"range": [70, 100], "color": "#ef4444"}
            ]
        }
    ))
    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}
    )
    st.plotly_chart(fig, use_container_width=True)

# ================= ECG UPLOAD =================
st.subheader("📤 Upload ECG Report Image")

uploaded_file = st.file_uploader(
    "Upload ECG Image (JPG / PNG)",
    type=["jpg", "png", "jpeg"]
)

ecg_status = "Unknown"
ecg_conf = 0

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    st.markdown("### 🫀 ECG Confirmation")
    confirm_ecg = st.checkbox(
        "I confirm that this uploaded image is a valid ECG report"
    )

    if confirm_ecg:
        ecg_status, ecg_conf = analyze_ecg_image(img)
        st.success(f"ECG Status: **{ecg_status}**")
        st.info(f"Confidence: **{ecg_conf}%**")
    else:
        st.warning("Please confirm that the uploaded image is an ECG report to proceed.")

# ================= INPUT FORM =================
with st.form("risk_form"):
    st.subheader("🫀 Patient & Medication Details")

    heart_rate = st.number_input("Heart Rate (BPM)", 20, 250, 72)
    chest_pain = st.checkbox("Chest Pain")
    shortness_breath = st.checkbox("Shortness of Breath")
    dizziness = st.checkbox("Dizziness")

    taking_drug = st.checkbox("Currently Taking Medication")
    dosage = st.selectbox("Dosage Level", ["Low", "Medium", "High"])

    age = st.number_input("Age", 0, 130, 45)
    history_hd = st.checkbox("History of Heart Disease")

    submitted = st.form_submit_button("🔮 Predict Cardiac Risk")

# ================= OUTPUT =================
if submitted:
    if ecg_status == "Unknown":
        st.warning("ECG-based analysis skipped because ECG confirmation was not provided.")

    risk = calculate_risk({
        "ecg": ecg_status,
        "hr": heart_rate,
        "pain": chest_pain,
        "breath": shortness_breath,
        "dizzy": dizziness,
        "drug": taking_drug,
        "dose": dosage,
        "age": age,
        "history": history_hd
    })

    st.subheader("📊 Cardiac Risk Assessment")
    show_speedometer(risk)

    st.subheader("🩺 Possible Heart-Related Conditions")
    diseases = predict_diseases({
        "ecg": ecg_status,
        "hr": heart_rate,
        "pain": chest_pain,
        "breath": shortness_breath,
        "history": history_hd
    }, risk)

    for d in diseases:
        st.markdown(f"- **{d}**")

    if risk >= 70:
        st.error("⚠️ High Cardiac Risk – Immediate medical consultation recommended")
    elif risk >= 40:
        st.warning("⚠️ Moderate Cardiac Risk – Monitoring advised")
    else:
        st.success("✅ Low Cardiac Risk")
