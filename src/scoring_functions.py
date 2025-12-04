import pandas as pd
import os
import numpy as np

def calculate_scores():
    input_path = os.path.join("data", "intermediate", "clean_data.csv")
    output_path = os.path.join("data", "features", "model_ready.csv")
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Standard Rates
    # ER_followers = engagements / followers
    # Handle division by zero or NaN followers
    # Ensure followers is numeric
    df['followers'] = pd.to_numeric(df['followers'], errors='coerce')
    df['engagements'] = pd.to_numeric(df['engagements'], errors='coerce')
    
    df['ER_followers'] = df['engagements'] / df['followers']
    
    # ER_impressions = engagements / impressions
    # Impressions missing, using followers as proxy as requested
    # We will just duplicate ER_followers for now or note it. 
    # User said: "If 'impressions' column is missing... use 'followers' as denominator and log this"
    df['ER_impressions'] = df['engagements'] / df['followers']
    
    # 2. Weighted Engagement Scores
    # Scheme A: Likes(1) + Comments(3) + Shares(5) (Updated per feedback)
    df['Scheme_A'] = df['likes_total'] * 1 + df['comments'] * 3 + df['shares'] * 5
    
    # Scheme B: Likes(1) + Comments(5) + Shares(3) (Updated per feedback)
    df['Scheme_B'] = df['likes_total'] * 1 + df['comments'] * 5 + df['shares'] * 3
    
    # Scheme C: (Likes(1) + Comments(3) + Shares(7)) [Normalized by Impressions/Followers]
    # Using followers as denominator since impressions missing
    weighted_sum_c = df['likes_total'] * 1 + df['comments'] * 3 + df['shares'] * 7
    df['Scheme_C'] = weighted_sum_c / df['followers']
    
    # 3. Time Decay Score
    # Calculate hours_since_publish
    # Reference time: Max post date in dataset
    df['post_date'] = pd.to_datetime(df['post_date'])
    max_date = df['post_date'].max()
    
    df['hours_since_publish'] = (max_date - df['post_date']).dt.total_seconds() / 3600
    
    # Formula: score * (1 / (1 + 0.1 * hours_since_publish))
    # Applying to Scheme A as the primary example, but let's create a generic decay factor
    # and maybe apply it to all schemes or just create a 'decayed_score' based on a default?
    # User said "Output: Apply these calculations".
    # I will create 'decay_factor' and 'decayed_Scheme_A', 'decayed_Scheme_B', 'decayed_Scheme_C'
    
    df['decay_factor'] = 1 / (1 + 0.1 * df['hours_since_publish'])
    
    df['decayed_Scheme_A'] = df['Scheme_A'] * df['decay_factor']
    df['decayed_Scheme_B'] = df['Scheme_B'] * df['decay_factor']
    df['decayed_Scheme_C'] = df['Scheme_C'] * df['decay_factor']
    
    # 4. Save
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Model ready data saved to {output_path}")
    print("Columns:", df.columns.tolist())
    print(df[['Scheme_A', 'Scheme_B', 'Scheme_C', 'decay_factor']].head())

if __name__ == "__main__":
    calculate_scores()
