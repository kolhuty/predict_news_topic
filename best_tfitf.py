import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
from nltk.stem.snowball import SnowballStemmer
import re
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Инициализация инструментов для обработки текста
nltk.download('stopwords')
russian_stop_words = stopwords.words('russian')
morph = MorphAnalyzer()
stemmer = SnowballStemmer("russian")

# Функции предобработки текста
def preprocess_text(row, lemmatize=True, stem=False):
    """Объединение, очистка и обработка текста"""

    combined = str(row).lower()
    combined = re.sub(r'[^а-яё\s]', '', combined)

    words = combined.split()
    words = [word for word in words if word not in russian_stop_words and len(word) > 2]

    if lemmatize:
        words = [morph.parse(word)[0].normal_form for word in words]
    elif stem:
        words = [stemmer.stem(word) for word in words]

    return ' '.join(words)

def tfitf_lem(X_train, y_train, X_test):
    # Обработка текстов
    print("Обработка текстов...")
    X_train_lemm = X_train.apply(preprocess_text, lemmatize=True, stem=False)
    X_test_lemm = X_test.apply(preprocess_text, lemmatize=True, stem=False)

    #Лемматизация + TF-IDF
    print("\nЛемматизация + TF-IDF...")
    tfidf = TfidfVectorizer(max_df=0.9, min_df=3)
    X_train_tfidf = tfidf.fit_transform(X_train_lemm)
    X_test_tfidf = tfidf.transform(X_test_lemm)

    model_multi = LogisticRegression(C=8.0, multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model_multi.fit(X_train_tfidf, y_train)

    return model_multi.predict(X_test_tfidf)


# Загрузка данных
data = pd.read_csv('lenta_archive.csv')
subm = pd.read_csv("base_submission_news.csv")
Test = pd.read_csv("test_news.csv")

#print(Test.head(5))
X = data['content']
y = data['topic']
subm['topic'] = tfitf_lem(X, y, Test['content'])
subm.to_csv("bow_logreg_lenta.csv", index=False)
