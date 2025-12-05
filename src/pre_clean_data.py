#Run this before anything else
#Source json corupts anysort of ai agent so need to quickly move away from the json.

import json
import csv
import os

# --- CONFIGURATION ---
# Assumes you put the json in data/raw/
# Script is now in src/, but we assume execution from project root
INPUT_FILE = os.path.join('data', 'raw', 'blazel_dataset_linkedin.json')
OUTPUT_FILE = os.path.join('data', 'intermediate', 'linkedin_data_cleaned.csv')

# Define the specific fields to keep
FIELDS_TO_KEEP = {
    "likes": ["numLikes"],
    "comments": ["numComments"],
    "shares": ["numShares"],
    "poll_votes": ["poll", "results", "uniqueVotersCount"],
    "post_text": ["text"],
    "video_duration": ["linkedinVideo", "videoPlayMetadata", "duration"],
    "doc_pages": ["document", "totalPageCount"],
    "poll_question": ["poll", "question"],
    "reshare_text": ["resharedPost", "text"],
    "author": ["authorName"],
    "followers": ["authorFollowersCount"],
    "post_date": ["postedAtISO"],
    "post_url": ["url"],
    "is_activity": ["isActivity"]
}

def get_nested_value(data, path_list):
    current = data
    for key in path_list:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current

def main():
    print(f"Looking for: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: File not found at {INPUT_FILE}")
        print("Please move your 'blazel_dataset_linkedin.json' into the 'data/raw/' folder.")
        return

    print(f"Processing...")

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDS_TO_KEEP.keys())
            writer.writeheader()

            count = 0
            for entry in raw_data:
                row = {}
                for col_name, path in FIELDS_TO_KEEP.items():
                    val = get_nested_value(entry, path)
                    
                    # --- CLEANING ---
                    if isinstance(val, str):
                        # Replace standard newlines/tabs
                        val = val.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
                        # Replace the "Unusual Line Terminators" (Unicode 2028/2029)
                        val = val.replace('\u2028', ' ').replace('\u2029', ' ')
                    
                    row[col_name] = val
                
                writer.writerow(row)
                count += 1

        print(f"Success! Created {OUTPUT_FILE}")
        print(f"Rows processed: {count}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
