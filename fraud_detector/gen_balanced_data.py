import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

def generate_balanced_data(n_samples=1000):
    # 50% Safe, 50% Fraud
    n_fraud = n_samples // 2
    n_safe = n_samples - n_fraud
    
    # Generate Safe Transactions: Low amounts, mostly day, rarely mismatch
    safe_data = {
        'amount': np.random.uniform(10, 2500, n_safe),
        'transaction_type': 'online', # Fixed to match current UI
        'location_mismatch': np.random.choice([0, 1], n_safe, p=[0.98, 0.02]),
        'time_of_day': np.random.choice([0, 1], n_safe, p=[0.9, 0.1]),
        'is_fraud': 0
    }
    
    # Generate Fraud Transactions: High amounts, mostly night, often mismatch
    fraud_data = {
        'amount': np.random.uniform(4000, 20000, n_fraud),
        'transaction_type': 'online', 
        'location_mismatch': np.random.choice([0, 1], n_fraud, p=[0.2, 0.8]),
        'time_of_day': np.random.choice([0, 1], n_fraud, p=[0.1, 0.9]),
        'is_fraud': 1
    }
    
    df_safe = pd.DataFrame(safe_data)
    df_fraud = pd.DataFrame(fraud_data)
    
    df = pd.concat([df_safe, df_fraud]).sample(frac=1).reset_index(drop=True)
    df.to_csv('sample_data.csv', index=False)
    print(f"Dataset generated with {n_samples} rows (Balanced 50/50 Fraud/Safe).")

if __name__ == "__main__":
    generate_balanced_data()
