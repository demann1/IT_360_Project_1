# utils/vectorizer.py

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.config import VECTORIZER_PATH, TFIDF_PARAMS

def build_vectorizer():
    return TfidfVectorizer(**TFIDF_PARAMS)

def fit_and_save_vectorizer(texts):
    vec = build_vectorizer()
    X = vec.fit_transform(texts)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vec, f)
    return X

def load_vectorizer():
    with open(VECTORIZER_PATH, 'rb') as f:
        return pickle.load(f)

def transform_text(texts):
    vec = load_vectorizer()
    return vec.transform(texts)
