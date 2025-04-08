# scripts/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from model.trainer import train_model, evaluate_model
from utils.vectorizer import fit_and_save_vectorizer
from utils.config import DATA_PATH

def load_data():
    df = pd.read_csv(DATA_PATH)
    texts = df['cleaned_text'].values
    labels = df['label'].values
    return train_test_split(texts, labels, test_size=0.2, random_state=42)

def main():
    # Load and split data
    X_train, X_test, y_train, y_test = load_data()
    
    # Vectorize text
    X_train_vec = fit_and_save_vectorizer(X_train)
    X_test_vec = transform_text(X_test)
    
    # Train model
    model = train_model(X_train_vec, y_train)
    
    # Evaluate
    evaluate_model(model, X_test_vec, y_test)

if __name__ == "__main__":
    main()