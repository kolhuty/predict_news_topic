import re
import nltk
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer

nltk.download('stopwords')
russian_stop_words = stopwords.words('russian')
morph = MorphAnalyzer()

def preprocess_text(text):
    """Cleaning and lemmatization of text"""
    text = str(text).lower()
    text = re.sub(r'[^а-яё\s]', '', text)

    words = text.split()
    words = [word for word in words if word not in russian_stop_words and len(word) > 2]
    words = [morph.parse(word)[0].normal_form for word in words]

    return ' '.join(words)
