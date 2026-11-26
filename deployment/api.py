"""House Price Estimator: prediction API (FastAPI).

Run locally:   uvicorn deployment.api:app --reload      (from the repository root)
Then open:     http://127.0.0.1:8000/docs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import PricePredictor

app = FastAPI(title='House Price Estimator API', version='1.0')
predictor = PricePredictor()


class Property(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    age: float
    location: str
    property_type: str


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(item: Property):
    try:
        return {'status': 'ok', **predictor.predict_price(**item.model_dump())}
    except ValueError as problem:
        raise HTTPException(status_code=422, detail=str(problem))
