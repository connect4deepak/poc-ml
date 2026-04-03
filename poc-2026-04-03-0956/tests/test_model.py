import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train a random forest classifier
clf = RandomForestClassifier(random_state=1)
clf.fit(X, y)

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Load the saved model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Make predictions using the loaded model
y_pred = loaded_model.predict(X)

# Evaluate the model
accuracy = loaded_model.score(X, y)
assert accuracy > 0.9