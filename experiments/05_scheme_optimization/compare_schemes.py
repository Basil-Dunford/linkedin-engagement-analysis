import pandas as pd
import os
import numpy as np
import scoring_functions

def compare_scoring_schemes():
    # 1. Provide a way to run the scoring update first
    print("Updating scores with new schemes...")
    scoring_functions.calculate_scores()
    
    # 2. Load the newly saved data
    input_path = os.path.join("data", "features", "model_ready.csv")
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    
    # 3. Define Target (Matching src/models.py logic)
    valid_er_count = df['ER_followers'].count()
    total_count = len(df)
    
    if valid_er_count < total_count * 0.1:
        print(f"WARNING: Too few valid ER_followers ({valid_er_count}/{total_count}). "
              "Falling back to raw 'engagements' for target definition.")
        target_metric = 'engagements'
        # Use full dataframe
        df[target_metric] = pd.to_numeric(df[target_metric], errors='coerce').fillna(0)
        validation_df = df
    else:
        target_metric = 'ER_followers'
        validation_df = df.dropna(subset=['ER_followers']).copy()
        
    threshold = validation_df[target_metric].quantile(0.8)
    validation_df['is_top_20'] = (validation_df[target_metric] >= threshold).astype(int)
    
    print(f"\nTarget Definition: Top 20% by {target_metric} (Threshold: {threshold:.4f})")
    print(f"Sample size for correlation: {len(validation_df)}")

    # 4. Identify all schemes
    # Look for columns starting with 'Scheme_'
    scheme_cols = [c for c in df.columns if c.startswith('Scheme_') and not c.startswith('Scheme_C_')] 
    # Note: Scheme_C is in there, but just filter generally
    
    predictions = {}
    
    print(f"\nComparing {len(scheme_cols)} schemes...")
    
    results = []
    
    for scheme in scheme_cols:
        # Correlation with Continuous Target defined above
        # Check if scheme column exists in validation_df (it should)
        if scheme not in validation_df.columns:
            continue
            
        corr_continuous = validation_df[scheme].corr(validation_df[target_metric])
        
        # Correlation with Binary Target (Point Biserial)
        corr_binary = validation_df[scheme].corr(validation_df['is_top_20'])
        
        results.append({
            'Scheme': scheme,
            'Corr_Continuous': corr_continuous,
            'Corr_Top20_Binary': corr_binary
        })
        
    # 5. Create DataFrame and Sort
    results_df = pd.DataFrame(results)
    
    # Sort by Binary Correlation (Primary Goal)
    results_df = results_df.sort_values('Corr_Top20_Binary', ascending=False).reset_index(drop=True)
    
    print("\n--- Scheme Comparison Results (Ranked by Top 20% Correlation) ---")
    print(results_df) # Print all
    
    # Comparison Summary
    best_scheme = results_df.iloc[0]
    baseline_scheme = results_df[results_df['Scheme'] == 'Scheme_Baseline'].iloc[0]
    original_c = results_df[results_df['Scheme'] == 'Scheme_C'].iloc[0]
    
    print("\n--- Summary ---")
    print(f"Best Scheme: {best_scheme['Scheme']} (Corr: {best_scheme['Corr_Top20_Binary']:.4f})")
    print(f"Original Best (C): {original_c['Corr_Top20_Binary']:.4f}")
    print(f"Baseline (1x1x1): {baseline_scheme['Corr_Top20_Binary']:.4f}")
    
    # Save results
    results_path = os.path.join("data", "scheme_comparison_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nFull results saved to {results_path}")

if __name__ == "__main__":
    compare_scoring_schemes()
