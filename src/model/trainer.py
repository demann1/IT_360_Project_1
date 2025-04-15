# model/trainer.py

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
try: 
    from src.utils.config import MODEL_PATH
except (ModuleNotFoundError, ImportError) as e:
    print (f"Error importing config: {e}")
    MODEL_PATH = 'model/logistic_regression_model.pkl'

def train_model(X, y):
    """Train and save a logistic regression model"""
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X, y)
    save_model(model)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    print("\n📊 Model Evaluation Report")
    print("="*40)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def save_model(model):
    """Save trained model to disk"""
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

def load_model():
    """Load trained model from disk"""
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)