from fastapi import FastAPI
from pydantic import BaseModel
import pickle

class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()
model = pickle.load(open('model.pkl', 'rb'))

@app.post('/predict')
def predict(iris: Iris):
    prediction = model.predict([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])
    return {'prediction': prediction[0]}