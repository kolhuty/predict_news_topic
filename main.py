import pandas as pd
from src.model import train_model

if __name__ == "__main__":
    data = pd.read_csv("data/lenta_archive.csv").sample(1000)
    X = data['content']
    y = data['topic']

    train_model(X, y)