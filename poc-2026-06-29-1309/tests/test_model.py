import pandas as pd
from train import model

# Load data
data = pd.read_csv('data.csv')

# Split data into features and target
X = data.drop('target', axis=1)
Y = data['target']

# Make predictions
predictions = model.predict(X)

# Evaluate model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y, predictions)
assert accuracy > 0.5