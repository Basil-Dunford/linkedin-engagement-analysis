import pandas as pd
import os
import numpy as np

def calculate_scores():
    # Note: Path adjusted for experiment folder (going up 2 levels)
    # Actually, scripts usually run from root. We will assume execution from root for simplicity or handle paths.
    input_path = os.path.join("data", "intermediate", "clean_data.csv")
    output_path = os.path.join("experiments", "05_scheme_optimization", "scored_data_v2.csv")
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return
        
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Preprocessing
    df['followers'] = pd.to_numeric(df['followers'], errors='coerce')
    df['engagements'] = pd.to_numeric(df['engagements'], errors='coerce').fillna(0)
    df['likes_total'] = pd.to_numeric(df['likes_total'], errors='coerce').fillna(0)
    df['comments'] = pd.to_numeric(df['comments'], errors='coerce').fillna(0)
    df['shares'] = pd.to_numeric(df['shares'], errors='coerce').fillna(0)
    
    # Calculate ER (Needed for target definition logic in compare_schemes)
    df['ER_followers'] = df['engagements'] / df['followers']
    
    # --- ABSOLUTE BEST SCHEME (Grid Search Result) ---
    # Optimized Weights: Likes=1, Comments=15, Shares=0
    df['Scheme_Optimized'] = df['likes_total'] * 1 + df['comments'] * 15 + df['shares'] * 0
    
    # Also include the other strong contenders for reference
    df['Scheme_Discussion'] = df['likes_total'] * 1 + df['comments'] * 15 + df['shares'] * 5
    df['Scheme_B'] = df['likes_total'] * 1 + df['comments'] * 5 + df['shares'] * 3
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Scored data saved to {output_path}")

if __name__ == "__main__":
    calculate_scores()
