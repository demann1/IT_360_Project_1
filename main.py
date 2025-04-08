# main.py

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import DATA_PATH
from src.utils.preprocessing import clean_text
from src.utils.vectorizer import fit_and_save_vectorizer, transform_text
from src.model.trainer import train_model, evaluate_model

def main():
    # Load the dataset
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please check the path.")

    # Check for required columns
    if 'v2' not in df.columns:
        raise KeyError("The column 'v2' (text data) is missing from the dataset.")
    if 'v1' not in df.columns:
        raise KeyError("The column 'v1' (target labels) is missing from the dataset.")

    # Clean the text data
    df['cleaned'] = df['v2'].apply(clean_text)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned'], df['v1'], test_size=0.2, random_state=42
    )

    # Vectorize the text data
    X_train_vec = fit_and_save_vectorizer(X_train)
    X_test_vec = transform_text(X_test)

    # Train the model
    model = train_model(X_train_vec, y_train)

    # Evaluate the model
    evaluate_model(model, X_test_vec, y_test)

    # Predict labels for the test set
    y_pred = model.predict(X_test_vec)

    # Add predicted labels and correctness to the DataFrame
    test_results = X_test.reset_index(drop=True).to_frame()
    test_results['actual_label'] = y_test.reset_index(drop=True)
    test_results['predicted_label'] = y_pred
    test_results['is_correct'] = test_results['actual_label'] == test_results['predicted_label']

    # Save the results to a new CSV file
    output_path = "test_results.csv"
    test_results.to_csv(output_path, index=False)
    print(f"Test results saved to {output_path}")

if __name__ == "__main__":
    main()
