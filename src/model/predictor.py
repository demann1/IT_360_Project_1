# model/predictor.py

from model.trainer import load_model
from utils.vectorizer import transform_text
from utils.preprocessing import clean_text

def predict_email(text):
    model = load_model()
    cleaned = clean_text(text)
    vectorized = transform_text([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction
