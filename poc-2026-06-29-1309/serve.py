from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

class PredictionRequest(BaseModel):
    features: list

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post('/predict')
def predict(request: PredictionRequest):
    prediction = model.predict(request.features)
    return {'prediction': prediction[0]}