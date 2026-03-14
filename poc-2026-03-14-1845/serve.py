from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post('/predict')
def predict(iris: Iris):
    prediction = model.predict([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])
    return {'prediction': prediction[0]}