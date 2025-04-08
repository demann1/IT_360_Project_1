# main.py

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config import DATA_PATH
from utils.preprocessing import clean_text
from utils.vectorizer import fit_and_save_vectorizer, transform_text
from model.trainer import train_model, evaluate_model

def main():
    df = pd.read_csv(DATA_PATH)
    df['cleaned'] = df['email_text'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['label'], test_size=0.2, random_state=42)

    X_train_vec = fit_and_save_vectorizer(X_train)
    X_test_vec = transform_text(X_test)

    model = train_model(X_train_vec, y_train)
    evaluate_model(model, X_test_vec, y_test)

if __name__ == "__main__":
    main()
