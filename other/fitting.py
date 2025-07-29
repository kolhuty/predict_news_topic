import torch
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump
from tqdm import tqdm
import pandas as pd
import warnings

# Игнорируем предупреждения о n_jobs
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


def train_and_save_model(data, test_data):

    def get_embeddings(texts, tokenizer, model, device, batch_size=32):
        assert isinstance(texts, list)
        assert all(isinstance(t, str) for t in texts), "Все элементы должны быть строками"

        model.eval()
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Извлечение эмбеддингов"):
            batch = texts[i:i + batch_size]

            encoded = tokenizer(
                batch,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=512,  # Должно совпадать с обучающей фазой
                return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                outputs = model(**encoded)
                # Используем тот же метод, что и при обучении ([CLS] токен)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    # Предобработка данных
    data['content'] = data['title'].fillna('') + ' ' + data['content'].fillna('')
    data = data[data['content'].notnull()].copy()
    data['content'] = data['content'].astype(str)
    data = data[data['content'].str.len() > 50]
    test_data = test_data[test_data['content'].notnull()].copy()
    test_data['content'] = test_data['content'].astype(str)
    #test_data = test_data[test_data['content'].str.len() > 50]
    print(f"Нормализация: {data['topic'].value_counts(normalize=True)}")

    logging.debug(f"Train samples: {len(data)} | Test samples: {len(test_data)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")

    # Загрузка РУССКОЙ модели и токенизатора
    model_name = 'cointegrated/rubert-tiny2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    X_train = get_embeddings(data['content'].tolist(), tokenizer, model, device)
    y_train = data['topic']
    X_test = get_embeddings(test_data['content'].tolist(), tokenizer, model, device)

    # Оптимизированный поиск параметров (RandomizedSearch вместо GridSearch)
    param_dist = {
        'C': [0.01, 0.1, 1, 3, 10],  # Уменьшенный набор параметров
        'solver': ['saga'],  # Только один solver, поддерживающий параллелизацию
        'class_weight': [None, 'balanced']
    }

    # Используем только 10 итераций и 3 фолда
    clf = RandomizedSearchCV(
        LogisticRegression(max_iter=1000, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring='accuracy',
        verbose=1,
        random_state=42
    )

    clf.fit(X_train, y_train)

    print(f"\nЛучшие параметры: {clf.best_params_}")
    print(f"Лучшая точность на валидации: {clf.best_score_:.4f}")

    # Оценка на тестовых данных
    best_model = clf.best_estimator_
    # test_acc = accuracy_score(y_test, best_model.predict(X_test))
    # train_acc = accuracy_score(y_train, best_model.predict(X_train))

    # print(f"\nТочность на обучении: {train_acc:.4f}")
    # print(f"Точность на тесте:    {test_acc:.4f}")

    # Сохранение лучшей модели
    dump(best_model, 'best_model.joblib')
    tokenizer.save_pretrained('tokenizer/')
    model.save_pretrained('bert/')

    test_output = pd.DataFrame({'content': best_model.predict(X_test)})

    return test_output


data = pd.read_csv('data.csv', encoding='utf-8')
test_data = pd.read_csv('test_news.csv', encoding='utf-8')

# Запуск функции
test_topic = train_and_save_model(data, test_data)
print(test_topic.head())
test_topic.to_csv('test_output.csv', index=True)