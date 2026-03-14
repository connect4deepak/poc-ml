import pytest
from sklearn.ensemble import RandomForestClassifier
import pickle

def test_model_accuracy():
    # Load iris dataset
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Split dataset into training set and test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # Create a classifier
    clf = RandomForestClassifier(random_state=1)
    # Train the classifier
    clf.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    # Evaluate the classifier
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.9

def test_model_prediction():
    # Load the saved model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Make a prediction
    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
    assert prediction[0] == 0