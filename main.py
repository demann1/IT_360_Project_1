# main.py

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import DATA_PATH
from src.utils.preprocessing import clean_text
from src.utils.vectorizer import fit_and_save_vectorizer, transform_text
from src.model.trainer import train_model, evaluate_model

def main():
    df = pd.read_csv(DATA_PATH)
    if 'v2' in df.columns:
        df['cleaned'] = df['v2'].apply(clean_text)
    else:
        raise KeyError("The column 'v2' is missing from the dataset.")

    X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['predicted_label'], test_size=0.2, random_state=42)

    X_train_vec = fit_and_save_vectorizer(X_train)
    X_test_vec = transform_text(X_test)

    model = train_model(X_train_vec, y_train)
    evaluate_model(model, X_test_vec, y_test)

if __name__ == "__main__":
    main()
