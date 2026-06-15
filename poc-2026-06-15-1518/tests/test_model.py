import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
accuracy = accuracy_score(y_test, y_pred)
assert accuracy > 0.7

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Load the model from the file
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Predict the response for test dataset using the loaded model
y_pred_loaded = loaded_model.predict(X_test)

# Model Accuracy: how often is the classifier correct?
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
assert accuracy_loaded > 0.7