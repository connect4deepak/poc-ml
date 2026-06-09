import pickle
import unittest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class TestModel(unittest.TestCase):
    def test_model_accuracy(self):
        dataset = load_iris()
        X = dataset.data
        y = dataset.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        self.assertGreater(accuracy_score(y_test, y_pred), 0.8)
    def test_model_pickle(self):
        clf = pickle.load(open('model.pkl', 'rb'))
        self.assertIsInstance(clf, RandomForestClassifier)