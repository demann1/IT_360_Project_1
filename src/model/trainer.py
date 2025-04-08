# model/trainer.py

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from utils.config import MODEL_PATH

def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("📊 Evaluation Report:\n")
    print(classification_report(y_test, y_pred))

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)
