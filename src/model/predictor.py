# model/predictor.py

from model.trainer import load_model
from utils.vectorizer import transform_text
from utils.preprocessing import clean_text

class PhishingDetector:
    def __init__(self):
        self.model = load_model()
    
    def predict_email(self, text):
        """Predict if an email is phishing"""
        cleaned = clean_text(text)
        vectorized = transform_text([cleaned])
        proba = self.model.predict_proba(vectorized)[0]
        prediction = self.model.predict(vectorized)[0]
        
        return {
            'is_phishing': bool(prediction),
            'phishing_probability': float(proba[1]),
            'text': cleaned
        }

def predict_from_file(file_path):
    """Predict from a file containing email text"""
    with open(file_path, 'r') as f:
        text = f.read()
    detector = PhishingDetector()
    return detector.predict_email(text)