import pandas as pd
from sklearn.model_selection import train_test_split

from utils.cleaner import clean_text
from utils.vectorizer import get_vectorizer
from model.trainer import train_model, evaluate_model

# Load data
df = pd.read_csv('data/emails.csv')
df['cleaned'] = df['email_text'].apply(clean_text)

# Vectorize
vectorizer = get_vectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train & evaluate
model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)
