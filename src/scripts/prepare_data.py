# scripts/prepare_data.py
import sys
from pathlib import Path

# Add the /src directory to the Python module search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from utils.preprocessing import clean_text

def prepare_dataset(input_path, output_path):
    # Load the test dataset
    df = pd.read_csv(input_path)
    
    # Rename columns for consistency
    df = df.rename(columns={'v2': 'text', 'v1': 'label'})
    
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Save processed data
    df[['cleaned_text', 'label']].to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    prepare_dataset('src/Dataset/test_email_dataset.csv', 'src/Dataset/emails.csv')