import unittest
from train import dataset, X, y, X_train, X_test, y_train, y_test
from sklearn.metrics import accuracy_score
import pickle

class TestModel(unittest.TestCase):
    def test_model_accuracy(self):
        model = pickle.load(open('model.pkl', 'rb'))
        y_pred = model.predict(X_test)
        self.assertGreater(accuracy_score(y_test, y_pred), 0.9)

if __name__ == '__main__':
    unittest.main()