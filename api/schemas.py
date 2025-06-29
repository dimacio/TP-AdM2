from pydantic import BaseModel, validator
from datetime import datetime
from enum import Enum # Importar desde el módulo 'enum'

# --- Usando Enum para crear constantes seguras ---

class StockColumn(str, Enum):
    """Define las columnas de datos de acciones como constantes seguras."""
    OPEN = "Open"
    HIGH = "High"
    LOW = "Low"
    CLOSE = "Close"
    VOLUME = "Volume"
    DIVIDENDS = "Dividends"
    STOCK_SPLITS = "Stock Splits"

class MetricType(str, Enum):
    """Define las métricas de evaluación como constantes seguras."""
    MSE = "mse"
    RMSE = "rmse"
    R2 = "r2"

# --- Modelos de Pydantic ---

class PredictInput(BaseModel):
    date: str = datetime.now().strftime('%Y-%m-%d')
    ticker: str = "NVDA"
    metric: MetricType = MetricType.RMSE # Usamos el Enum para el valor por defecto

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
    # El dag_id ya no es dinámico, se define en el DAG
    dag_id: str = 'stock_prediction_pipeline'
    ticker: str = "NVDA"
