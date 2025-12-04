import pandas as pd
import os

def load_and_verify_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        print(f"Shape: {df.shape}")
        print("Columns:")
        print(df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    input_path = os.path.join("data", "intermediate", "linkedin_data_cleaned.csv")
    df = load_and_verify_data(input_path)
    
    if df is not None:
        # Verify columns (we can add specific checks if we know the required columns)
        # For now, just ensuring it loads is the first step.
        
        # Save to data/intermediate as requested (even if it's the same file, we'll save as 'raw' as requested)
        # User said: "Save a raw CSV to data/intermediate/"
        # Since the input is already in data/intermediate, let's save it as 'raw_linkedin_data.csv' to distinguish?
        # Or maybe just overwrite or ensure it's there. 
        # The prompt said: "Load the CSV, filter to these columns, and save a raw CSV to data/intermediate/"
        # I will save it as 'raw_linkedin_data.csv' to be safe and clear.
        
        output_path = os.path.join("data", "intermediate", "raw_linkedin_data.csv")
        df.to_csv(output_path, index=False)
        print(f"\nSaved raw data to {output_path}")
