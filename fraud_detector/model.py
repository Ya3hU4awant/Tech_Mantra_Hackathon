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

    # 3. Synchronize Prediction & Risk Level Logic
    # We will now derive EVERYTHING from the probability of fraud.
    is_fraud_pred = (pred_label == 1)
    # PyCaret 'prediction_score' is the probability of the predicted class.
    # Convert to probability of fraud (label 1).
    prob_fraud = confidence if is_fraud_pred else (1 - confidence)

    # Determine Risk Level based on probability_of_fraud
    if prob_fraud >= THRESHOLD_HIGH:
        risk_level = "High"
    elif prob_fraud >= THRESHOLD_MEDIUM:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    prediction_text = "Fraud" if is_fraud_pred else "Safe"
    
    # Deep Analysis explanation
    explanation_parts = []
    if float(input_data.get('amount', 0)) > 5000: explanation_parts.append("High amount")
    if input_data.get('location_mismatch') == 'yes': explanation_parts.append("Location mismatch")
    if input_data.get('time_of_day') == 'night': explanation_parts.append("Night signature")

    if is_fraud_pred:
        explanation = f"Detected pattern matching known fraud signatures ({int(prob_fraud*100)}% risk). { ' + '.join(explanation_parts) if explanation_parts else '' }"
    else:
        explanation = f"Transaction appears secure. Risk evaluation: {int(prob_fraud*100)}%."

    feature_distributions = [
        {"label": "Digital Transactions", "value": 85 if input_data.get('transaction_type') == 'online' else 15, "color": "#3498db"},
        {"label": "In-Store Swipe", "value": 85 if input_data.get('transaction_type') == 'in-store' else 15, "color": "#3498db"},
        {"label": "Wire Transfer", "value": 85 if input_data.get('transaction_type') == 'wire-transfer' else 15, "color": "#3498db"},
        {"label": "Location Mismatch", "value": 80 if input_data.get('location_mismatch') == 'yes' else 10, "color": "#e67e22"},
        {"label": "Night Transaction", "value": 90 if input_data.get('time_of_day') == 'night' else 10, "color": "#9b59b6"},
        {"label": "Transaction Amount", "value": min(100, int(float(input_data.get('amount', 0)) / 100)), "color": "#16a085"}
    ]

    return {
        "prediction": prediction_text,
        "risk_level": risk_level,
        "explanation": explanation,
        "confidence": confidence,
        "prob_fraud": prob_fraud,
        "distributions": feature_distributions
    }
