# fastapi backend for HHAR Activity Recognition API

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

app = FastAPI(title="HHAR Activity Recognition API")

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "encoder.pkl"))

class SensorInput(BaseModel):
    x: float
    y: float
    z: float

@app.get("/")
def home():
    return {"message": "HHAR Activity Recognition API is Running!"}

@app.post("/predict")
def predict(data: SensorInput):

    prediction = model.predict([[data.x, data.y, data.z]])

    activity = encoder.inverse_transform(prediction)[0]

    return {
        "activity": activity
    }