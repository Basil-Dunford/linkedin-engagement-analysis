import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

def visualize_analysis():
    # Paths
    data_path = os.path.join("data", "features", "model_ready.csv")
    model_path = os.path.join("data", "models", "lgbm_model.pkl")
    viz_dir = os.path.join("data", "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Loading model from {model_path}...")
    clf = joblib.load(model_path)
    
    # Define features used in the model (must match training)
    feature_cols = [
        'weekday', 'hour', 'word_count', 'has_emoji', 'has_hashtag',
        'video_duration', 'doc_pages'
    ]
    media_cols = [c for c in df.columns if c.startswith('media_type_')]
    feature_cols.extend(media_cols)
    
    # Prepare X for SHAP
    X = df[feature_cols].fillna(0)
    
    # 1. SHAP Dependence Plots
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    
    # Handle SHAP output format (lightgbm binary returns list or array depending on version)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # Positive class
        
    # Features to analyze (Importance > 0 based on previous analysis)
    target_features = [
        'word_count', 'hour', 'weekday', 'video_duration', 
        'has_emoji', 'has_hashtag', 'media_type_Text'
    ]
    
    print("Generating SHAP dependence plots...")
    for feature in target_features:
        if feature in X.columns:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature, shap_values, X, show=False)
            plt.title(f"SHAP Dependence: {feature}")
            out_path = os.path.join(viz_dir, f"shap_dependence_{feature}.png")
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
            print(f"Saved {out_path}")
            
    # 2. Binning Analysis
    print("Performing Binning Analysis...")
    # Ensure engagements is numeric
    df['engagements'] = pd.to_numeric(df['engagements'], errors='coerce').fillna(0)
    
    for feature in target_features:
        if feature not in df.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        
        if feature in ['word_count', 'video_duration']:
            # Continuous: Bin into quantiles or fixed ranges
            # Use qcut for equal frequency bins, or cut for equal width. 
            # qcut is usually better for skewed data like word_count
            try:
                df[f'{feature}_bin'] = pd.qcut(df[feature], q=10, duplicates='drop')
            except ValueError:
                # Fallback if not enough unique values
                df[f'{feature}_bin'] = pd.cut(df[feature], bins=5)
                
            sns.barplot(x=f'{feature}_bin', y='engagements', data=df, errorbar=None)
            plt.xticks(rotation=45)
            plt.xlabel(f"{feature} Range")
        else:
            # Discrete/Categorical
            sns.barplot(x=feature, y='engagements', data=df, errorbar=None)
            plt.xlabel(feature)
            
        plt.ylabel("Average Engagements")
        plt.title(f"Average Engagements by {feature}")
        out_path = os.path.join(viz_dir, f"binning_{feature}.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_path}")

if __name__ == "__main__":
    visualize_analysis()
