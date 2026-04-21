import unittest
from train import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class TestModel(unittest.TestCase):
    def test_model(self):
        # Load iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        # Train model
        model = RandomForestClassifier(random_state=1)
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        self.assertGreater(accuracy, 0.9)