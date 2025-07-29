from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.preprocess import preprocess_text

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    print("Text preprocessing...")
    X_train_proc = X_train.apply(preprocess_text)
    X_test_proc = X_test.apply(preprocess_text)

    tfidf = TfidfVectorizer(max_df=0.9, min_df=3)
    X_train_tfidf = tfidf.fit_transform(X_train_proc)
    X_test_tfidf = tfidf.transform(X_test_proc)

    model = LogisticRegression(C=8.0, solver='lbfgs', max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_pred, y_test)

    print(f"Accuracy: {acc:.4f}")
    return model, tfidf
