import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib

def train_model():
    input_path = os.path.join("data", "features", "model_ready.csv")
    model_dir = os.path.join("data", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Define Target
    # Top 20% of ER_followers
    # Handle NaNs in ER_followers just in case
    df['ER_followers'] = pd.to_numeric(df['ER_followers'], errors='coerce')
    
    valid_er_count = df['ER_followers'].count()
    print(f"Valid ER_followers count: {valid_er_count}")
    
    if valid_er_count < len(df) * 0.1: # If less than 10% have valid ER
        print("WARNING: Too few valid ER_followers. Falling back to raw 'engagements' for target definition.")
        target_metric = 'engagements'
        df[target_metric] = pd.to_numeric(df[target_metric], errors='coerce').fillna(0)
    else:
        target_metric = 'ER_followers'
        df[target_metric] = df[target_metric].fillna(0)
        
    threshold = df[target_metric].quantile(0.8)
    df['is_high_performing'] = (df[target_metric] >= threshold).astype(int)
    
    print(f"Target Metric: {target_metric}")
    print(f"High performance threshold: {threshold}")
    print(f"Class balance:\n{df['is_high_performing'].value_counts(normalize=True)}")
    
    # 2. Define Features
    # Content features only (exclude outcome metrics like likes, comments, shares, schemes)
    feature_cols = [
        'weekday', 'hour', 'word_count', 'has_emoji', 'has_hashtag',
        'video_duration', 'doc_pages'
    ]
    
    # Add media_type columns
    media_cols = [c for c in df.columns if c.startswith('media_type_')]
    feature_cols.extend(media_cols)
    
    # Fill NaNs in features
    df[feature_cols] = df[feature_cols].fillna(0)
    
    X = df[feature_cols]
    y = df['is_high_performing']
    
    # 3. Time-based Split
    # Sort by post_date
    df['post_date'] = pd.to_datetime(df['post_date'], format='mixed')
    df = df.sort_values('post_date')
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 4. Train LightGBM
    print("Training LightGBM model...")
    clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
    clf.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Save model
    model_path = os.path.join(model_dir, "lgbm_model.pkl")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
    
    # 6. SHAP Values
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    
    # Handle SHAP output format (lightgbm binary returns list or array depending on version)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # Positive class
        
    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    
    # Save to visualizations directory
    viz_dir = os.path.join("data", "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    shap_plot_path = os.path.join(viz_dir, "shap_summary.png")
    
    plt.savefig(shap_plot_path, bbox_inches='tight')
    print(f"SHAP summary plot saved to {shap_plot_path}")
    
    # Save feature importance to CSV for report
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    importance_path = os.path.join(model_dir, "feature_importance.csv")
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")
    print("\nTop 5 Features:")
    print(feature_importance.head())

if __name__ == "__main__":
    train_model()
