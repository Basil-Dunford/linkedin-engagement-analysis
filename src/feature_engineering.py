import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re

def analyze_nlp():
    input_path = os.path.join("data", "features", "model_ready.csv")
    output_dir = os.path.join("data", "features")
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Re-define target (same logic as models.py to ensure consistency)
    # We need to know which posts are "High Performing" to correlate
    df['ER_followers'] = pd.to_numeric(df['ER_followers'], errors='coerce')
    valid_er_count = df['ER_followers'].count()
    
    if valid_er_count < len(df) * 0.1:
        target_metric = 'engagements'
        df[target_metric] = pd.to_numeric(df[target_metric], errors='coerce').fillna(0)
    else:
        target_metric = 'ER_followers'
        df[target_metric] = df[target_metric].fillna(0)
        
    threshold = df[target_metric].quantile(0.8)
    df['is_high_performing'] = (df[target_metric] >= threshold).astype(int)
    
    print(f"Target Metric for NLP: {target_metric}")
    
    # 1. Linguistic Patterns
    def check_patterns(text):
        text = str(text).lower()
        return pd.Series({
            'is_question': 1 if '?' in text else 0,
            'is_statement': 0 if '?' in text else 1, # Simplified
            'first_person': 1 if re.search(r'\b(i|me|my)\b', text) else 0,
            'collective': 1 if re.search(r'\b(we|us|our)\b', text) else 0
        })
        
    patterns = df['post_text'].apply(check_patterns)
    df = pd.concat([df, patterns], axis=1)
    
    # Correlate patterns with high performance
    correlations = patterns.apply(lambda x: x.corr(df['is_high_performing']))
    print("\nLinguistic Pattern Correlations with High Performance:")
    print(correlations.sort_values(ascending=False))
    
    # Save correlations
    correlations.to_csv(os.path.join(output_dir, "nlp_correlations.csv"))
    
    # 2. TF-IDF Analysis
    print("\nRunning TF-IDF Analysis...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_matrix = tfidf.fit_transform(df['post_text'].fillna(''))
    feature_names = tfidf.get_feature_names_out()
    
    # Average TF-IDF score for High vs Low performing
    high_perf_idx = df['is_high_performing'] == 1
    low_perf_idx = df['is_high_performing'] == 0
    
    # Convert to numpy array for sparse matrix indexing
    avg_tfidf_high = np.array(tfidf_matrix[high_perf_idx.values].mean(axis=0)).flatten()
    avg_tfidf_low = np.array(tfidf_matrix[low_perf_idx.values].mean(axis=0)).flatten()
    
    diff_tfidf = avg_tfidf_high - avg_tfidf_low
    
    # Top keywords for High Performing
    top_indices = diff_tfidf.argsort()[-20:][::-1]
    top_keywords = [(feature_names[i], diff_tfidf[i]) for i in top_indices]
    
    print("\nTop 20 Keywords associated with High Performance (High - Low TF-IDF):")
    for kw, score in top_keywords:
        print(f"{kw}: {score:.4f}")
        
    # Save keywords
    pd.DataFrame(top_keywords, columns=['keyword', 'score']).to_csv(os.path.join(output_dir, "top_keywords.csv"), index=False)

if __name__ == "__main__":
    analyze_nlp()
