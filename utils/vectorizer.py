from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

def get_vectorizer():
    return TfidfVectorizer(stop_words=stopwords.words('english'), max_features=3000)
