import pandas as pd
import os
import numpy as np

def analyze():
    input_path = os.path.join("data", "intermediate", "clean_data.csv")
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    
    # Preprocessing (same as scoring_functions.py)
    df['followers'] = pd.to_numeric(df['followers'], errors='coerce')
    df['engagements'] = pd.to_numeric(df['engagements'], errors='coerce').fillna(0)
    df['likes_total'] = pd.to_numeric(df['likes_total'], errors='coerce').fillna(0)
    df['comments'] = pd.to_numeric(df['comments'], errors='coerce').fillna(0)
    df['shares'] = pd.to_numeric(df['shares'], errors='coerce').fillna(0)
    
    # Calculate ER
    df['ER_followers'] = df['engagements'] / df['followers']
    
    # Schemes
    df['Scheme_A'] = df['likes_total'] * 1 + df['comments'] * 3 + df['shares'] * 5
    df['Scheme_B'] = df['likes_total'] * 1 + df['comments'] * 5 + df['shares'] * 3
    # Scheme C (normalized)
    weighted_sum_c = df['likes_total'] * 1 + df['comments'] * 3 + df['shares'] * 7
    df['Scheme_C'] = weighted_sum_c / df['followers']
    
    # Target: Top 20% of ER_followers
    # Filter valid ER
    valid_er = df.dropna(subset=['ER_followers'])
    threshold = valid_er['ER_followers'].quantile(0.8)
    valid_er['is_top_20'] = (valid_er['ER_followers'] >= threshold).astype(int)
    
    print(f"Top 20% Threshold (ER): {threshold}")
    
    schemes = ['Scheme_A', 'Scheme_B', 'Scheme_C']
    
    print("\n--- Correlations with ER_followers (Continuous) ---")
    print(valid_er[schemes + ['ER_followers']].corr()['ER_followers'])
    
    print("\n--- Correlations with is_top_20 (Point Biserial / Pearson) ---")
    print(valid_er[schemes + ['is_top_20']].corr()['is_top_20'])

if __name__ == "__main__":
    analyze()
