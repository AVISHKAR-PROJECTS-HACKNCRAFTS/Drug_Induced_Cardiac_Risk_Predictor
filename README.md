# 🫀 AI Cardiac Risk Prediction System

An advanced **AI-powered Cardiac Risk Prediction System** leveraging **Deep Learning (BiLSTM)** and Clinical Data Analysis to provide real-time cardiac health assessments.

This application combines **ECG Signal Analysis** with patient-specific clinical factors (medication, symptoms, history) to accurately predict cardiac risk levels and identify potential conditions.

---

## 🚀 Key Features

- 📤 **Intelligent ECG Analysis** — Upload standard ECG images (PNG/JPG) for instant AI processing.
- 🧠 **Deep Learning Core** — Powered by a **Bidirectional LSTM (Long Short-Term Memory)** network with **Attention Mechanism**.
- 📊 **High Accuracy** — The model achieves **93.60% Test Accuracy** on the MIT-BIH Arrhythmia Database.
- 💊 **Drug-Induced Risk Engine** — Calculates risk scores based on dosage, drug interactions, and potential side effects.
- 🩺 **Clinical Decision Support** — Predicts specific conditions like **Arrhythmia**, **Tachycardia**, **Bradycardia**, and **Coronary Artery Disease (CAD)**.
- 📈 **Modern Medical Dashboard** — Features real-time risk gauges, confidence scores, and a premium dark-mode UI.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python Package Installer)

### Step 1: Install Dependencies
Open your terminal or command prompt in the project folder and run:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application
Start the local server by running:
```bash
streamlit run app.py
```

### Step 3: Access the Dashboard
The application will automatically open in your default web browser at:
> **http://localhost:8501**

---

## 📖 User Guide

1.  **Upload ECG**: Drag & Drop your ECG report image into the upload section.
2.  **Verify**: Check the "This is a valid ECG report" box to confirm the image is clear.
3.  **AI Analysis**: The system will extract the waveform and classify it as **Normal** or **Abnormal** with a confidence score.
4.  **Patient Data**: Fill in the clinical form (Age, Heart Rate, Symptoms, Medication).
5.  **Get Results**: Click **"Analyze Cardiac Risk"** to generate the comprehensive risk report.

---

## 🧠 Model Architecture

The system uses a custom **PyTorch Deep Learning Model** designed specifically for time-series ECG data:

- **Architecture:** Bidirectional LSTM with Attention
- **Why BiLSTM?** It processes the ECG signal in both forward and backward directions, capturing temporal dependencies that standard models miss.
- **Attention Mechanism:** Automatically focuses on the most critical parts of the heartbeat (like the QRS complex) for higher precision.
- **Training Data:** Trained on **67,662 balanced heartbeat segments** from the MIT-BIH database.

**Performance Metrics:**
| Metric | Score |
| :--- | :--- |
| **Accuracy** | **93.60%** |
| **F1 Score** | **93.46%** |
| **AUC-ROC** | **0.9571** |

---

## � Dataset Information

### MIT-BIH Arrhythmia Database
The model was trained using the industry-standard **MIT-BIH Arrhythmia Database** from PhysioNet.

- **Source:** Beth Israel Hospital (Boston) & MIT
- **Content:** 48 half-hour excerpts of two-channel ambulatory ECG recordings.
- **Subjects:** 47 subjects studied by the BIH Arrhythmia Laboratory.
- **Sampling:** 360 Hz per channel with 11-bit resolution.
- ** preprocessing:** Signals were segmented, denoised, and balanced to ensure the model detects abnormal beats as accurately as normal ones.

---

## 📁 Project Structure

```
CARDIAC_RISK_APP/
├── app.py                       # Main Application (Streamlit + Model Inference)
├── requirements.txt             # List of required Python libraries
├── README.md                    # Project Documentation
└── saved_model/                 # Trained AI Model Files
    ├── cardiac_lstm_model.pth   # PyTorch Model Weights (The "Brain")
    ├── scaler.pkl               # Data Preprocessing Scaler
    ├── label_classes.pkl        # Class Definitions (Normal/Abnormal)
    ├── model_architecture.json  # Model Configuration
    └── training_metrics.json    # Final Training Performance Logs
```

---

## ⚠️ Disclaimer

This tool is designed for **assistive and educational purposes only**. It is **not** a substitute for professional medical diagnosis. All predictions should be verified by a qualified cardiologist or healthcare professional.
