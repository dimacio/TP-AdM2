# Stock Prediction API

Estando en este directorio (/api) correr los siguientes comandos:

```
docker build -t stock-predictor-api .
docker run -p 8000:8000 stock-predictor-api
```

Verificar que la api esta corriendo accediendo a:

```
http://localhost:8000/docs
```

# Estructura de carpetas

```
api/
├── main.py # Only FastAPI app, endpoints, and dependency injection
├── services/
│ ├── mlflow_service.py # MLflow/model/scaler logic
│ ├── stock_service.py # yfinance/data logic
│ └── auth_service.py # Auth/JWT/Redis logic
├── schemas.py # Pydantic models
```
