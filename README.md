# LinkedIn Engagement Analysis

## Project Overview
This project analyzes LinkedIn post performance to identify drivers of high engagement. It uses a pipeline of data cleaning, feature engineering, and predictive modeling (LightGBM) to generate actionable insights.

## Project Structure
- `data/`: Contains raw, intermediate, and feature data.
    - `raw/`: Raw JSON input.
    - `intermediate/`: Cleaned CSVs.
    - `features/`: Model-ready data and NLP outputs.
    - `models/`: Trained LightGBM model and SHAP plots.
- `src/`: Source code.
    - `ingest_data.py`: Loads and verifies raw data.
    - `data_cleaning.py`: Cleans data and extracts features.
    - `scoring_functions.py`: Calculates engagement scores (Schemes A, B, C).
    - `models.py`: Trains predictive model and computes SHAP values.
    - `feature_engineering.py`: Performs NLP analysis (TF-IDF, linguistic patterns).
- `docs/`: Documentation and reports.
    - `final_report.md`: Detailed findings and recommendations.
    - `project_log.md`: Log of data assumptions and issues.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your raw JSON data in `data/raw/`.

## Usage
Run the pipeline in order:

1. **Ingest Data**:
   ```bash
   python src/ingest_data.py
   ```
2. **Clean Data**:
   ```bash
   python src/data_cleaning.py
   ```
3. **Calculate Scores**:
   ```bash
   python src/scoring_functions.py
   ```
4. **Train Model**:
   ```bash
   python src/models.py
   ```
5. **Run NLP Analysis**:
   ```bash
   python src/feature_engineering.py
   ```

## Key Findings
- **Scheme B** (Likes + 5*Comments + 3*Shares) is the best predictor of success.
- **Top Features**: Word count, posting hour, and video duration.
- **Actionable Tip**: Use "We/Us" language to build community.
