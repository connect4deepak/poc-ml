from fastapi import FastAPI
from pydantic import BaseModel
import pickle

class IrisDataset(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()
model = pickle.load(open('model.pkl', 'rb'))

@app.post('/predict')
def predict(data: IrisDataset):
    prediction = model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    return {'prediction': prediction[0]}