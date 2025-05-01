# main.py

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import DATA_PATH
from src.utils.preprocessing import clean_text
from src.utils.vectorizer import fit_and_save_vectorizer, transform_text
from src.model.trainer import train_model, evaluate_model
from src.api.api_caller import fetch_emails
from src.utils.preprocessing import clean_text
from src.utils.vectorizer import transform_text
from src.model.trainer import load_model

def classify_emails(email_texts):
    # Clean the emails
    cleaned_emails = [clean_text(email) for email in email_texts]

    # Load the vectorizer and transform the emails
    vectorized_emails = transform_text(cleaned_emails)

    # Load the trained model
    model = load_model()

    # Predict labels
    predictions = model.predict(vectorized_emails)
    return predictions

def summarize_results(sample_data):
    """Generate a summary of the dataset and predictions."""
    total_emails = len(sample_data)
    spam_count = sample_data['predicted_label'].sum()
    non_spam_count = total_emails - spam_count

    # Calculate accuracy if 'is_correct' column exists
    if 'is_correct' in sample_data.columns and not sample_data['is_correct'].isnull().all():
        accuracy = sample_data['is_correct'].mean() * 100
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("Accuracy: Not available (missing 'is_correct' data)")

    print(f"Total Emails: {total_emails}")
    print(f"Spam Emails: {spam_count}")
    print(f"Non-Spam Emails: {non_spam_count}")

def main(local_run=True):
    if local_run:
        print("Running in local mode...")
        # Fetch emails
        email_texts = fetch_emails(max_results=30)

        # Classify the emails
        predictions = classify_emails(email_texts)

        # Create a DataFrame for the fetched emails
        sample_data = pd.DataFrame({
            'email_text': email_texts,
            'predicted_label': predictions
        })

        # Print the results
        for email, prediction in zip(email_texts, predictions):
            label = "Spam" if prediction == 1 else "Not Spam"
            print(f"Email: {email}\nPrediction: {label}\n{'-' * 40}")

        # Generate and print the summary
        summarize_results(sample_data)

    else:
        print("Running in dataset mode...")
        # Load the sample dataset
        dataset_path = "src/dataset/test_email_dataset.csv"
        try:
            sample_data = pd.read_csv(dataset_path)
        except FileNotFoundError:
            print(f"Dataset not found at {dataset_path}. Please check the path.")
            return

        # Check if the required column exists
        if 'email_text' not in sample_data.columns:
            print("The dataset must contain an 'email_text' column.")
            return

        # Extract email texts
        email_texts = sample_data['email_text'].tolist()

        # Classify the emails
        predictions = classify_emails(email_texts)

        # Add predictions to the dataset
        sample_data['predicted_label'] = predictions

        # Check correctness by comparing predictions with actual labels
        if 'v1' in sample_data.columns:
            sample_data['is_correct'] = sample_data['v1'] == sample_data['predicted_label']
        else:
            print("The dataset must contain a 'v1' column for actual labels.")
            return

        # Print the results
        for email, prediction in zip(email_texts, predictions):
            label = "Spam" if prediction == 1 else "Not Spam"
            print(f"Email: {email}\nPrediction: {label}\n{'-' * 40}")

        # Generate and print the summary
        summarize_results(sample_data)

        # Save the updated dataset with predictions and correctness
        output_path = "test_results.csv"
        sample_data.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the email classification script.")
    parser.add_argument(
        "--local", 
        action="store_true", 
        help="Run the script in local mode (fetch and classify emails)."
    )
    args = parser.parse_args()

    # Determine the mode based on the --local flag
    main(local_run=args.local)
