# El propósito del siguiente script puede ser:
# -Evaluar modelos ya entrenados en otras condiciones.
# -Comparar modelos guardados contra nuevos datos.
# -Usar otro contenedor para simular un modelo ya deployado en producción.

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.keras
import tensorflow as tf
import yfinance as yf
import os
from datetime import datetime

# --- Parámetros ---
TICKER = os.getenv("TICKER", "AAPL")
START_DATE = os.getenv("START_DATE", "2020-01-01")
LOOKBACK_WINDOW = 60
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Evaluacion_Modelo_LSTM")
MLFLOW_S3_BUCKET_NAME = os.getenv("MLFLOW_BUCKET_NAME", "mi-bucket-experimentos")
MODEL_ARTIFACT_PATH = f"s3://{MLFLOW_S3_BUCKET_NAME}/artifacts/Prediccion_Acciones_LSTM_DC_Final_KerasAutoLog/model/data/model.keras"

FEATURES_COLUMNS = ['Open', 'High', 'Low', 'Volume']
TARGET_COLUMN = 'Close'

# --- Cargar y preparar datos ---
def cargar_datos(ticker):
    end_date = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=START_DATE, end=end_date)
    df = df.dropna(subset=FEATURES_COLUMNS + [TARGET_COLUMN])
    df = df[FEATURES_COLUMNS + [TARGET_COLUMN]]
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaled_x = scaler_x.fit_transform(df[FEATURES_COLUMNS])
    scaled_y = scaler_y.fit_transform(df[[TARGET_COLUMN]])
    scaled_data = np.concatenate((scaled_x, scaled_y), axis=1)
    return df.index, scaled_data, scaler_y

def crear_secuencias(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :])
        y.append(data[i, -1])
    return np.array(X), np.array(y)

# --- MLflow Setup ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

with mlflow.start_run() as run:
    print(f"Evaluando modelo para ticker: {TICKER}")
    
    # Cargar modelo
    print("Cargando modelo desde S3...")
    model = tf.keras.models.load_model(MODEL_ARTIFACT_PATH)

    # Datos
    fechas, data, scaler_y = cargar_datos(TICKER)
    X, y = crear_secuencias(data, LOOKBACK_WINDOW)
    if len(X) == 0:
        print("No hay datos suficientes para evaluar.")
        mlflow.end_run(status="FAILED")
        exit()

    # Evaluar
    loss = model.evaluate(X, y, verbose=0)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)

    # Log de métricas
    mlflow.log_param("ticker", TICKER)
    mlflow.log_metric("eval_loss_mse", loss)
    mlflow.log_metric("eval_mse", mse)
    mlflow.log_metric("eval_rmse", rmse)
    mlflow.log_metric("eval_r2", r2)

    print(f"Métricas:\nLoss: {loss:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}")
