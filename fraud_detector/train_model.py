"""
train_model.py
Section: Model Training for Financial Fraud Detection
Description: This script uses PyCaret to automatically compare multiple machine learning 
classifiers and save the best performing model for later use in our Flask app.
"""

import pandas as pd
from pycaret.classification import setup, compare_models, finalize_model, save_model

# 1. Load the generated dataset
DATA_PATH = 'sample_data.csv'
MODEL_NAME = 'fraud_model'

def train_best_model():
    print(f"--- Starting Model Training with {DATA_PATH} ---")
    
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    
    # 2. Setup the experiment
    # fix_imbalance=True handles the 14.5% fraud rate using SMOTE
    exp = setup(
        data=df, 
        target='is_fraud', 
        session_id=42,
        fix_imbalance=True,
        fix_imbalance_method='smote',
        log_experiment=False,
        verbose=False,
        html=False
    )
    
    # 3. Compare multiple models and select the best one
    # This will print a table of metrics (Accuracy, AUC, Recall, etc.)
    print("Comparing models... This may take a minute.")
    best_model = compare_models()
    
    print("\n--- Best Model Selection ---")
    print(best_model)
    
    # 4. Finalize the model (train on entire dataset)
    final_model = finalize_model(best_model)
    
    # 5. Save the model as .pkl
    save_model(final_model, MODEL_NAME)
    print(f"\nModel saved successfully as {MODEL_NAME}.pkl")

if __name__ == "__main__":
    try:
        train_best_model()
    except Exception as e:
        print(f"Error during training: {e}")
        print("Note: Ensure 'pycaret' and 'pandas' are installed: pip install pycaret pandas")
