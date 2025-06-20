from pydantic import BaseModel, validator
from datetime import datetime
from typing import Literal

StockColumn = Literal['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
MetricType = Literal['mse', 'rmse', 'r2']

class PredictInput(BaseModel):
    date: str = datetime.now().strftime('%Y-%m-%d')  # Format: 'YYYY-MM-DD'
    ticker: str = "NVDA"
    metric: MetricType = "rmse"  # Default metric for model selection

    @validator('date')
    def date_must_be_yyyy_mm_dd(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('date must be in YYYY-MM-DD format')
        return v

class LoginInput(BaseModel):
    username: str
    password: str

class StockInput(BaseModel):
    open: float
    high: float
    low: float
    volume: float

class DagRunInput(BaseModel):
    dag_id: str = 'taskflow_stock_prediction_pipeline'