import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
from sklearn.metrics import classification_report, roc_auc_score
# Import the isolated scoring function
import scoring_v2

def train_model():
    print("Step 1: Updating scores via scoring_v2...")
    scoring_v2.calculate_scores()
    
    input_path = os.path.join("experiments", "05_scheme_optimization", "scored_data_v2.csv")
    if not os.path.exists(input_path):
        print(f"Data not found: {input_path}")
        return
        
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Define Target: Top 20% by Scheme_Optimized
    # The user asked to "train a new model on the new optimized scheme".
    # This implies using Scheme_Optimized as the ground truth for "High Performance".
    
    target_metric = 'Scheme_Optimized'
    threshold = df[target_metric].quantile(0.8)
    df['is_high_performing'] = (df[target_metric] >= threshold).astype(int)
    
    print(f"Target: Top 20% by {target_metric} (Threshold: {threshold:.2f})")
    
    # Features (Same as original model)
    feature_cols = [
        'weekday', 'hour', 'word_count', 'has_emoji', 'has_hashtag',
        'video_duration', 'doc_pages'
    ]
    media_cols = [c for c in df.columns if c.startswith('media_type_')]
    feature_cols.extend(media_cols)
    
    X = df[feature_cols].fillna(0)
    y = df['is_high_performing']
    
    # Time Split
    df['post_date'] = pd.to_datetime(df['post_date'], format='mixed')
    df = df.sort_values('post_date')
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train
    print("Training LightGBM...")
    clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Save Model
    model_dir = os.path.dirname(input_path)
    model_path = os.path.join(model_dir, "model_optimized.pkl")
    joblib.dump(clf, model_path)
    print(f"Optimized Model saved to {model_path}")

    # --- SHAP Analysis ---
    import shap
    import matplotlib.pyplot as plt

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    
    # Handle SHAP output format (lightgbm binary returns list or array depending on version)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # Positive class
        
    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    
    shap_plot_path = os.path.join(model_dir, "shap_summary_optimized.png")
    plt.savefig(shap_plot_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to {shap_plot_path}")

if __name__ == "__main__":
    train_model()
