# 🛡️ Financial Fraud Detector - AI/ML Hackathon Starter

This is a complete, beginner-friendly Flask-based project designed to detect financial fraud patterns in digital transactions using Machine Learning (PyCaret).

## 🚀 Project Overview
The "Machine Learning Model for Early Detection of Financial Fraud Patterns in Digital Transactions" identifies suspicious activities based on transaction amount, geographical consistency, and timing. It features a premium dashboard aesthetic with real-time risk assessment and AI-generated explanations.

## 📁 Project Structure
```text
fraud_detector/
├── app.py              # Flask Web Server
├── model.py            # Inference logic (Loads model & predicts)
├── train_model.py      # ML Training Script (Uses PyCaret)
├── sample_data.csv     # Dummy dataset (200 rows)
├── README.md           # Documentation
├── templates/
│   ├── index.html      # Analysis Form
│   └── result.html     # Prediction Results
└── static/
    └── style.css       # Custom Premium Styling
```

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.9 or higher** installed.

### 2. Install Dependencies
Run the following command to install the required libraries:
```bash
pip install flask pandas pycaret
```

### 3. Generate the ML Model
Before running the application, you must train the model using the provided sample data:
```bash
python train_model.py
```
*This will analyze several classifiers and save the best one as `fraud_model.pkl`.*

### 4. Run the Application
Start the Flask server:
```bash
python app.py
```
Open your browser and navigate to: `http://localhost:8000`

## 🧠 How it Works
1. **Data:** A synthetic dataset of 200 transactions is used, where high-risk patterns (e.g., high amounts at night with location mismatch) are labeled as fraud.
2. **ML Engine:** PyCaret's `compare_models()` automatically tests various algorithms (Random Forest, XGBoost, Logistic Regression, etc.) and picks the one with the highest accuracy/AUC.
3. **Inference:** When you submit the form, the data is preprocessed and passed to the saved model.
4. **Logic:** The `risk_level` is derived from the model's probability score, and an `explanation` is generated based on the specific features that triggered the flag.

## 📈 Extending the Project
- **Real Data:** Replace `sample_data.csv` with a dataset from Kaggle (e.g., Credit Card Fraud Detection).
- **Advanced Features:** Add more inputs like `device_fingerprint`, `merchant_category`, or `account_age`.
- **API Integration:** Connect the backend to a real payment gateway callback.

---
*Created for AI/ML Hackathon participants.*
