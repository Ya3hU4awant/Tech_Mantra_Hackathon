"""
model.py
Section: Inference Logic for Fraud Detection
Description: Contains logic to load the pre-trained model and provide predictions
with human-readable explanations.
"""

import pandas as pd
from pycaret.classification import load_model, predict_model

import os

# Constants for risk levels
THRESHOLD_HIGH = 0.7
THRESHOLD_MEDIUM = 0.4

MODEL_NAME = 'fraud_model'
# Get the directory where model.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)

# Pre-load the model when the module is imported
try:
    # PyCaret's load_model adds .pkl automatically
    model = load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Could not load model at '{MODEL_PATH}.pkl'. Error: {e}")

def predict_fraud(input_data: dict) -> dict:
    """
    Takes raw dictionary input and returns prediction results.
    input_data keys: 'amount', 'transaction_type', 'location_mismatch', 'time_of_day'
    """
    if model is None:
        return {
            "prediction": "Error",
            "risk_level": "N/A",
            "explanation": "Model file not found. Please train the model first.",
            "confidence": 0.0
        }

    # Convert dictionary to DataFrame for PyCaret
    # Ensure values are converted to correct types
    df_input = pd.DataFrame([{
        'amount': float(input_data.get('amount', 0)),
        'transaction_type': input_data.get('transaction_type', 'online'),
        'location_mismatch': 1 if input_data.get('location_mismatch') == 'yes' else 0,
        'time_of_day': 1 if input_data.get('time_of_day') == 'night' else 0
    }])

    # Get prediction and probability
    # 'prediction_label' and 'prediction_score' are returned by PyCaret 3.x
    predictions = predict_model(model, data=df_input)
    
    pred_label = int(predictions['prediction_label'].iloc[0])
    confidence = float(predictions['prediction_score'].iloc[0])

    # Derive human-readable results
    prediction_text = "Fraud" if pred_label == 1 else "Safe"
    
    # Calculate Risk Level based on probability
    if confidence >= THRESHOLD_HIGH:
        risk_level = "High"
    elif confidence >= THRESHOLD_MEDIUM:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Generate Explanation logic based on input features
    explanation_parts = []
    if float(input_data.get('amount', 0)) > 5000:
        explanation_parts.append("High transaction amount")
    if input_data.get('location_mismatch') == 'yes':
        explanation_parts.append("Mismatch in geographical location")
    if input_data.get('time_of_day') == 'night':
        explanation_parts.append("Transaction occurring at night")

    if not explanation_parts:
        explanation = "Transaction indicators are normal."
    else:
        status_word = "suspicious" if pred_label == 1 else "noteworthy"
        explanation = f"{' + '.join(explanation_parts)} flagged as {status_word}."

    return {
        "prediction": prediction_text,
        "risk_level": risk_level,
        "explanation": explanation,
        "confidence": round(confidence, 4)
    }
