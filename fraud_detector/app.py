"""
app.py
Section: Flask Web Server for Fraud Detection
Description: Main entry point that routes web requests and handles prediction calls.
"""

from flask import Flask, render_template, request, flash, redirect, url_for
from model import predict_fraud
import os

app = Flask(__name__)
app.secret_key = "fraud_detection_secret_key" # Required for flash messages

# Define Constants
APP_TITLE = "Financial Fraud Detector"

@app.route('/')
def index():
    """Renders the landing page with the input form."""
    return render_template('index.html', title=APP_TITLE)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles form submission, processes inputs, calling model.py for prediction,
    and returns result.html with the findings.
    """
    try:
        # 1. Extract and validate form data
        form_data = {
            'amount': request.form.get('amount'),
            'transaction_type': request.form.get('transaction_type'),
            'location_mismatch': request.form.get('location_mismatch'),
            'time_of_day': request.form.get('time_of_day')
        }

        # Check for missing values
        if not all(form_data.values()):
            flash("All fields are required. Please fill in everything.", "danger")
            return redirect(url_for('index'))

        # 2. Convert types and call ML logic
        # predict_fraud returns: prediction, risk_level, explanation, confidence
        results = predict_fraud(form_data)
        
        # 3. Render result template with all values
        return render_template(
            'result.html',
            title=APP_TITLE,
            input=form_data,
            results=results
        )

    except Exception as e:
        flash(f"An error occurred during prediction: {str(e)}", "danger")
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Start the Flask app
    # Debug mode is enabled for development
    print(f"Starting {APP_TITLE}...")
    app.run(debug=True, port=8000)
