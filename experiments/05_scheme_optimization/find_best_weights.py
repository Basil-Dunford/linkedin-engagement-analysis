import pandas as pd
import numpy as np
import os

def find_best_weights():
    # 1. Load Data
    input_path = os.path.join("data", "intermediate", "clean_data.csv")
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    
    # 2. Preprocess
    # Fill NaNs
    df['likes_total'] = pd.to_numeric(df['likes_total'], errors='coerce').fillna(0)
    df['comments'] = pd.to_numeric(df['comments'], errors='coerce').fillna(0)
    df['shares'] = pd.to_numeric(df['shares'], errors='coerce').fillna(0)
    df['engagements'] = pd.to_numeric(df['engagements'], errors='coerce').fillna(0)
    
    # 3. Define Target (Raw Engagements, Top 20%)
    target_metric = 'engagements'
    threshold = df[target_metric].quantile(0.8)
    df['is_top_20'] = (df[target_metric] >= threshold).astype(int)
    
    print(f"Target: Top 20% of {target_metric} (Threshold: {threshold})")
    print(f"Data shape: {df.shape}")
    
    # 4. Grid Search
    # Weights for Likes fixed at 1.
    # varies Comments and Shares from 0 to 100.
    
    best_corr = -1
    best_weights = (1, 1, 1) # L, C, S
    
    # Vectors for fast calculation
    L = df['likes_total'].values
    C = df['comments'].values
    S = df['shares'].values
    Y = df['is_top_20'].values
    
    print("Starting Grid Search (Weights 0-200)...")
    
    results = []
    
    # Using steps of 5 for speed/coverage, then we can refine if needed
    # User said "much higher weights", so let's go up to 200.
    weight_range = range(0, 201, 5) 
    
    total_combinations = len(weight_range) * len(weight_range)
    count = 0
    
    for wc in weight_range:
        for ws in weight_range:
            # Calculate Score
            # Avoid all zeros
            if wc == 0 and ws == 0:
                continue
                
            score = L * 1 + C * wc + S * ws
            
            # Correlation
            # We can use np.corrcoef but it returns a matrix. 
            # Or just pd.Series.corr which is slower but convenient.
            # Using numpy for speed:
            # Pearson correlation
            
            non_zero_score_mask = score != 0 # Optional: handled by correlation? 
            # Actually standard pearson handles it fine unless std is 0.
            
            # Fast binary correlation (Point Biserial is just Pearson with one binary var)
            corr = np.corrcoef(score, Y)[0, 1]
            
            if np.isnan(corr):
                continue
                
            if corr > best_corr:
                best_corr = corr
                best_weights = (1, wc, ws)
            
            # Store some data for distribution analysis
            if count % 500 == 0:
                pass # print(f"Scanned {count}/{total_combinations}...")
            count += 1
            
    print("\n--- Optimization Results ---")
    print(f"Best Correlation: {best_corr:.5f}")
    print(f"Best Weights: Likes=1, Comments={best_weights[1]}, Shares={best_weights[2]}")
    
    # Refined Search around the best?
    # If best is at the edge (e.g. 200), we might need to go higher.
    # If best is (1, 85, 20), we could check 80-90.
    # For now, this coarse grain is likely enough to answer "what is the best scheme".
    
    print(f"\nWinning Formula: 1 * Likes + {best_weights[1]} * Comments + {best_weights[2]} * Shares")

if __name__ == "__main__":
    find_best_weights()
