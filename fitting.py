import pandas as pd
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from joblib import dump, load

def model(data):

    # Проверим устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загрузка модели и токенизатора
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model = model.to(device)

    # Токенизация и преобразование входов
    encoded = tokenizer.batch_encode_plus(
        data['content'].tolist(),
        add_special_tokens=True,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:, 0, :].numpy()
    labels = data['topic']

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state=42)

    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)

    dump(lr_clf, 'model.joblib')  # сохраняем модель
    dump(tokenizer, 'vectorizer.joblib')

    return accuracy_score(train_labels, lr_clf.predict(train_features)), accuracy_score(test_labels,
                                                                                 lr_clf.predict(test_features))

