import pandas as pd
import os
import re
import numpy as np

def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text)

def count_words(text):
    return len(clean_text(text).split())

def has_emoji(text):
    # Basic emoji detection (this is a simplified regex, might need a library for full coverage but standard for quick check)
    # Using a range that covers many emojis
    return bool(re.search(r'[^\x00-\x7F]', clean_text(text)))

def has_hashtag(text):
    return bool(re.search(r'#\w+', clean_text(text)))

def infer_media_type(row):
    # Priority: Video > Document > Poll > Text (Image/Carousel hard to distinguish without specific cols)
    if pd.notna(row.get('video_duration')) and row['video_duration'] > 0:
        return 'Video'
    if pd.notna(row.get('doc_pages')) and row['doc_pages'] > 0:
        return 'Document'
    if pd.notna(row.get('poll_question')):
        return 'Poll'
    # Default to Text (which includes Images/Carousels if not distinguishable)
    return 'Text'

def clean_data():
    input_path = os.path.join("data", "intermediate", "raw_linkedin_data.csv")
    output_path = os.path.join("data", "intermediate", "clean_data.csv")
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Normalize Reactions
    # User confirmed: likes_total = likes
    df['likes_total'] = df['likes']
    
    # 2. Total Engagement
    # engagements = likes_total + comments + shares + clicks (clicks unavailable)
    df['engagements'] = df['likes_total'] + df['comments'] + df['shares']
    
    # 3. Features
    # Post Date parsing
    df['post_date'] = pd.to_datetime(df['post_date'])
    
    # Weekday (0=Monday, 6=Sunday)
    df['weekday'] = df['post_date'].dt.weekday
    
    # Hour (0-23)
    df['hour'] = df['post_date'].dt.hour
    
    # Text features
    df['word_count'] = df['post_text'].apply(count_words)
    df['has_emoji'] = df['post_text'].apply(has_emoji).astype(int)
    df['has_hashtag'] = df['post_text'].apply(has_hashtag).astype(int)
    
    # 4. Media Type
    df['media_type'] = df.apply(infer_media_type, axis=1)
    
    # One-hot encoding for media_type
    # Requested: text, image, video, carousel, document.
    # We have: Video, Document, Poll, Text.
    # We will map Poll to Text or keep it? User requested specific one-hots.
    # Let's keep what we found and maybe add placeholders if strictly needed, but better to reflect reality.
    # I will one-hot encode the inferred types.
    media_dummies = pd.get_dummies(df['media_type'], prefix='media_type')
    df = pd.concat([df, media_dummies], axis=1)
    
    # Ensure all requested columns exist even if 0 (for consistency if model expects them)
    expected_types = ['Text', 'Image', 'Video', 'Carousel', 'Document']
    for m_type in expected_types:
        col_name = f'media_type_{m_type}'
        if col_name not in df.columns:
            df[col_name] = 0
            
    # 5. Save
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print("Columns:", df.columns.tolist())
    print(df[['media_type', 'engagements', 'weekday', 'hour']].head())

if __name__ == "__main__":
    clean_data()
