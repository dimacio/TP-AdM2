from fastapi import FastAPI, HTTPException, status, Depends, Form
import requests
from services.mlflow_service import (
    load_model_and_scaler, 
    get_best_run_id, 
    get_training_run_id_from_eval_run
)
from services.auth_service import create_access_token, verify_jwt_token
from datetime import datetime, timedelta
import redis
import yfinance as yf
import logging
import numpy as np
from schemas import MetricType, PredictInput, DagRunInput, StockColumn

# Configuración de logging para ver todo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Inicialización de Clientes y App ---
redis_client = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)
airflow_api_host = "http://airflow-webserver:8080"
app = FastAPI()
N_SAMPLES = 60

# --- Funciones Auxiliares ---
def get_last_n_ticker_prices(ticker: str, input_end_date: str, column: str, n: int = N_SAMPLES):
    log.info(f"Obteniendo precios para '{ticker}', columna '{column}'")
    target_date = datetime.strptime(input_end_date, '%Y-%m-%d')
    end_date = target_date.strftime('%Y-%m-%d')
    start_date = (target_date - timedelta(days=int(n*2.0))).strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron datos para el ticker {ticker}.")
    
    # Este es el punto exacto del error anterior
    last_n_ticker = df[column].dropna().tail(n).values
    
    if len(last_n_ticker) < n:
        raise HTTPException(status_code=400, detail=f"No hay suficientes datos históricos para el ticker {ticker} para predecir (se necesitan {n}, se encontraron {len(last_n_ticker)}).")
    return last_n_ticker

# --- Endpoints de la API ---
@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/login")
def login(username: str = Form(), password: str = Form()):
    # (sin cambios)
    url = f"{airflow_api_host}/api/v1/dags"
    try:
        resp = requests.get(url, auth=(username, password), timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        log.error(f"Error de autenticación con Airflow: {e}")
        raise HTTPException(status_code=500, detail=f"Falló la autenticación con Airflow: {e}")
    redis_client.setex(f"airflow:{username}:password", 3600, password)
    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/trigger-new-dag-run")
def trigger_new_dag_run(data: DagRunInput, username: str = Depends(verify_jwt_token)):
    # (sin cambios)
    password = redis_client.get(f"airflow:{username}:password")
    if not password:
        raise HTTPException(status_code=401, detail="Sesión expirada. Por favor, vuelve a iniciar sesión.")
    url = f"{airflow_api_host}/api/v1/dags/{data.dag_id}/dagRuns"
    payload = {"conf": {"ticker": data.ticker}}
    try:
        resp = requests.post(url, json=payload, auth=(username, password), timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Airflow API error: {e.response.text}")

@app.post("/predict-sample")
def predict_sample(input: PredictInput, username: str = Depends(verify_jwt_token)):
    try:
        log.info(f"Buscando el mejor run en el experimento de evaluación: Stock_Prediction_Evaluation_TaskFlow")
        eval_run_id = get_best_run_id("Stock_Prediction_Evaluation_TaskFlow", metric=input.metric)
        
        log.info(f"Mejor run de evaluación encontrado: {eval_run_id}. Buscando su run de entrenamiento padre...")
        training_run_id = get_training_run_id_from_eval_run(eval_run_id)
        
        model, scaler = load_model_and_scaler(training_run_id)
        
        column_to_fetch = StockColumn.OPEN.value
        log.info(f"Llamando a get_last_n_ticker_prices con la columna: '{column_to_fetch}' (Tipo: {type(column_to_fetch)})")
        
        last_n_samples_open = get_last_n_ticker_prices(input.ticker, input.date, column_to_fetch)
        
        features_scaled = scaler.transform(last_n_samples_open.reshape(-1, 1))
        X_pred = features_scaled.reshape(1, N_SAMPLES, 1)
        
        pred_scaled = model.predict(X_pred)
        pred = scaler.inverse_transform(pred_scaled)[0][0]
        
        log.info(f"Predicción para {input.ticker} generada exitosamente: {pred}")
        return {"ticker": input.ticker, "date": input.date, "prediction": float(pred)}

    except ValueError as e:
        log.error(f"Error de valor durante la predicción: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        log.error(f"Error inesperado en predicción: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Ocurrió un error interno al procesar la predicción.")
