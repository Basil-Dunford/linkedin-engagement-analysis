# Project Log

## Data Issues and Assumptions

### Missing Impressions Column
- **Issue**: The `impressions` column is missing from the source dataset (`linkedin_data_cleaned.csv`).
- **Resolution**: As per user instructions, `followers` is used as a proxy denominator for calculating `ER_impressions` and `Scheme_C`.
- **Impact**: Engagement rates normalized by impressions will essentially be engagement rates normalized by followers. This may underestimate the true engagement rate if impressions are significantly lower than follower count (which is typical).

### Media Type Inference
- **Issue**: No explicit `media_type` column.
- **Resolution**: Inferred media types based on available columns:
    - `video_duration` > 0 -> **Video**
    - `doc_pages` > 0 -> **Document**
    - `poll_question` exists -> **Poll**
    - Default -> **Text** (includes Images/Carousels/Text-only)
