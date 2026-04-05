import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

def test_model_accuracy():
    dataset = load_iris()
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.7

def test_model_pickle():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    assert isinstance(model, LogisticRegression)