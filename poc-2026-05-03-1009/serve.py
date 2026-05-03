from fastapi import FastAPI
from pydantic import BaseModel
import pickle

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post('/predict')
def predict(data: IrisData):
    prediction = model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    return {'prediction': prediction[0]}