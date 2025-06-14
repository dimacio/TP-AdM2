from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class StockInput(BaseModel):
    open: float
    high: float
    low: float
    volume: float

@app.post("/predict")
def predict(data: StockInput):
    return {"prediction": 42}  # Dummy response for testing
