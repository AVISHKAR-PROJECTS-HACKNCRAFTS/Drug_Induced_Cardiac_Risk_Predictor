🫀 Drug-Induced Cardiac Risk Prediction System (Streamlit App)

This project presents an AI-assisted Cardiac Risk Prediction System designed to support early identification of potential heart-related complications, especially those influenced by medication usage. The system combines ECG report image analysis, patient clinical details, and medication information to estimate future cardiac risk in an explainable and ethical manner.

The application is built using Streamlit and focuses on decision support and risk awareness, not medical diagnosis.

🚀 Key Features

📤 Upload ECG report images (PNG / JPG)

✅ User confirmation to ensure responsible ECG analysis

🫀 ECG rhythm classification (Normal / Irregular) with confidence score

📊 Cardiac risk estimation displayed using a speedometer gauge

🩺 Prediction of possible heart-related conditions (e.g., Arrhythmia, CAD)

💊 Incorporates drug usage, dosage, and patient symptoms

⚖️ Explainable, frontend-focused, and ethically designed system

🧠 How the System Works

User uploads an ECG report image

User confirms that the image is a valid ECG report

The system analyzes ECG waveform smoothness to classify rhythm

Clinical inputs (heart rate, symptoms, medication details, age, history) are collected

A cardiac risk score is calculated

Results are visualized and explained in a user-friendly format

⚠️ This system provides indicative results for awareness and does not replace professional medical diagnosis.

🖥️ Running the Application Locally
Prerequisites

Python 3.8 or higher

pip installed

Installation & Run
pip install -r requirements.txt
streamlit run app.py


The application will open in your browser.

📊 Dataset Information

Due to GitHub size limitations, the dataset is not included in this repository.

Dataset Name

MIT-BIH Arrhythmia Database

Official Source

🔗 https://physionet.org/content/mitdb/1.0.0/

Dataset Purpose in This Project

The MIT-BIH Arrhythmia Database is used as a reference dataset to justify ECG rhythm patterns and cardiac abnormalities. The current application does not directly display or preprocess dataset contents but aligns its ECG analysis logic with standard arrhythmia characteristics described in the dataset.

How to Add the Dataset (Optional)

Download the dataset from the official source

Extract the files

Place the extracted folder inside the project directory

mit-bih-arrhythmia-database-1.0.0/


The application will silently detect the dataset folder for internal calibration without exposing dataset data in the UI.

📌 Project Scope

Early cardiac risk awareness

Drug-induced cardiac complication support

Explainable AI-based decision support

Educational and research-oriented healthcare application

🔮 Future Enhancements

Integration of CNN/LSTM deep learning models for ECG classification

Automatic ECG vs non-ECG image validation

Real-time ECG signal analysis from wearable devices

Cloud-based deployment and patient history tracking

Multi-class cardiac disease prediction

📚 References

Moody, G. B., & Mark, R. G., The MIT-BIH Arrhythmia Database, IEEE

Goldberger, A. L., et al., PhysioNet: Components of a New Research Resource

Rajpurkar, P., et al., Cardiologist-Level Arrhythmia Detection Using Deep Learning

🧾 Disclaimer

This application is intended only for educational and research purposes.
It is not a medical diagnostic tool.