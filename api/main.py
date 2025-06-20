from fastapi import FastAPI, HTTPException, status
import requests
import jwt
from services.mlflow_service import load_model_and_scaler, get_best_run_id
from datetime import datetime, timedelta
from fastapi import Depends, Header
import redis
from datetime import datetime
from mlflow.tracking import MlflowClient
import yfinance as yf
import logging
from schemas import MetricType, PredictInput, DagRunInput, StockColumn
logging.basicConfig(level=logging.INFO)


redis_client = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)


airflow_api_host = "http://airflow-webserver:8080"

app = FastAPI()

SECRET_KEY = "your-very-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

N_SAMPLES=60


def get_last_n_ticker_prices(ticker: str, input_end_date: str, column: StockColumn, n: int = 60):    
    target_date = datetime.strptime(input_end_date, '%Y-%m-%d')
    end_date = target_date.strftime('%Y-%m-%d')
    start_date = (target_date - timedelta(days=int(n*1.5))).strftime('%Y-%m-%d')  # 90 days to ensure 60 trading days
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    last_n_ticker = df[column].dropna().values[-n:]

    if len(last_n_ticker) < n:
        raise HTTPException(status_code=400, detail="No hay suficientes datos para predecir.")

    return last_n_ticker


@app.get("/")
def health_check():
    return {"status": "ok"}


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_jwt_token(token: str = Header(...)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        # Optionally check Redis for token revocation
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/login")
def login(username: str, password: str):
    # Authenticate against Airflow using Basic Auth
    url = f"{airflow_api_host}/api/v1/dags"
    try:
        resp = requests.get(url, auth=(username, password), timeout=5)
        print(f"Airflow response: {resp.status_code} - {resp.text}")
        if resp.status_code != 200:
           raise HTTPException(status_code=resp.status_code, detail=f"Airflow API error: {resp.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Airflow authentication failed: {str(e)}")

    # Store password in Redis with TTL
    redis_client.setex(f"airflow:{username}:password", 3600, password)
    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/trigger-new-dag-run")
def trigger_new_dag_run(
    data: DagRunInput,
    username: str = Depends(verify_jwt_token)
):
    password = redis_client.get(f"airflow:{username}:password")
    if not password:
        raise HTTPException(status_code=401, detail="Session expired, please log in again.")
    
    url = f"{airflow_api_host}/api/v1/dags/{data.dag_id}/dagRuns"
    payload = {}

    try:
        resp = requests.post(url, json=payload, auth=(username, password))
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as e:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Airflow API error: {resp.text}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/get-best-model-run-id")
def get_best_model_run_id(
    experiment_name: str = "Stock_Prediction_Evaluation_TaskFlow",
    metric: MetricType = "rmse",
    ascending: bool = True
):
    try:
        run_id = get_best_run_id(experiment_name, metric, ascending)
        return {"run_id": run_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict-sample")
def predict_sample(input: PredictInput):
    run_id = get_best_run_id("Stock_Prediction_Evaluation_TaskFlow", metric=input.metric, ascending=True)
    logging.info(f"Using run_id: {run_id} for prediction")

    client = MlflowClient()
    artifacts = client.list_artifacts(run_id)
    logging.info(f"Artifacts for run {run_id}: {[a.path for a in artifacts]}")

    # 1. Load model and scaler
    model, scaler = load_model_and_scaler(run_id)

    # Get last 60 days of 'Open' prices before the given date
    last_n_samples_open = get_last_n_ticker_prices(input.ticker, input.date, 'Open', N_SAMPLES)

    features_scaled = scaler.transform(last_n_samples_open.reshape(-1, 1))
    X_pred = features_scaled.reshape(1, N_SAMPLES, 1)

    pred_scaled = model.predict(X_pred)
    pred = scaler.inverse_transform(pred_scaled)[0][0]

    return {
        "ticker": input.ticker,
        "date": input.date,
        "prediction": float(pred)
    }