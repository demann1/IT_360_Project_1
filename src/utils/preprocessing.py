# utils/preprocessing.py

import re
from urllib.parse import urlparse

def clean_text(text):
    """Clean and normalize email text"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '[URL]', text)
    
    # Remove special characters except basic punctuation
    text = re.sub(r"[^\w\s.!?]", ' ', text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", ' ', text).strip()
    
    return text

def extract_url_features(text):
    """Extract features from URLs in text"""
    urls = re.findall(r"http\S+|www\S+|https\S+", text)
    features = {
        'num_urls': len(urls),
        'has_url': int(len(urls) > 0)
    }
    
    if urls:
        # Analyze the first URL
        try:
            domain = urlparse(urls[0]).netloc
            features['url_length'] = len(urls[0])
            features['domain_has_ip'] = int(bool(re.match(r"\d+\.\d+\.\d+\.\d+", domain)))
        except:
            pass
    
    return features