import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('data.csv')

# Split data into features and target
X = data.drop('target', axis=1)
Y = data['target']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(Y_test, predictions)
print(f'Model accuracy: {accuracy:.3f}')

# Save model
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)