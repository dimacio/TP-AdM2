import os
import pendulum
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import mlflow
import mlflow.keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import logging

from airflow.decorators import dag, task

# Constantes del Pipeline
TRAINING_EXPERIMENT_NAME = "Stock_Prediction_Training"
# CORRECCIÓN: Asegurar que el nombre del experimento de evaluación coincida con el que busca la API
EVALUATION_EXPERIMENT_NAME = "Stock_Prediction_Evaluation_TaskFlow"

@dag(
    # CORRECCIÓN: El ID del DAG debe coincidir con el que la API llama
    dag_id="stock_prediction_pipeline",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    schedule=None,
    tags=["ml", "taskflow", "stocks"],
    params={"ticker": "NVDA"},
    is_paused_upon_creation=False
)
def stock_prediction_pipeline():

    @task
    def train_model(ticker: str, experiment_name: str) -> str:
        """
        Tarea que descarga datos, entrena un modelo LSTM y lo registra en MLflow.
        Retorna el run_id de MLflow para la siguiente tarea.
        """
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logging.info(f"--- Iniciando Tarea de Entrenamiento para {ticker} --- Run ID: {run_id}")
            
            start_date = "2010-01-01"
            end_date = datetime.today().strftime('%Y-%m-%d')
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty: raise ValueError(f"No se pudieron descargar datos para {ticker}.")

            data_training = data[data.index < '2022-01-01'].copy()
            training_set = data_training[['Open']].values
            
            sc = MinMaxScaler(feature_range=(0, 1))
            training_set_scaled = sc.fit_transform(training_set)
            
            X_train, y_train = [], []
            for i in range(60, len(training_set_scaled)):
                X_train.append(training_set_scaled[i-60:i, 0])
                y_train.append(training_set_scaled[i, 0])
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            
            regressor = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)), Dropout(0.2),
                LSTM(units=50), Dropout(0.2),
                Dense(units=1)
            ])
            
            regressor.compile(optimizer='adam', loss='mean_squared_error')
            logging.info("Entrenando el modelo...")
            regressor.fit(X_train, y_train, epochs=2, batch_size=32, verbose=1)
            
            scaler_path = "/tmp/scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(sc, f)
            
            logging.info("Guardando modelo y scaler en MLflow...")
            mlflow.keras.log_model(regressor, "model")
            mlflow.log_artifact(scaler_path, "scaler_artifact")
            mlflow.log_param("ticker", ticker)
            
            logging.info(f"--- Entrenamiento completado. Run ID: {run_id} ---")
            return run_id

    @task
    def evaluate_model(training_run_id: str, ticker: str, eval_experiment_name: str):
        logging.info(f"--- Iniciando Tarea de Evaluación para {ticker} ---")
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        
        model_uri = f"runs:/{training_run_id}/model"
        logging.info(f"Cargando modelo desde: {model_uri}")
        loaded_model = mlflow.keras.load_model(model_uri)
        
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(training_run_id, "scaler_artifact")
        scaler_path = os.path.join(local_path, "scaler.pkl")
        with open(scaler_path, "rb") as f:
            loaded_scaler = pickle.load(f)
        
        start_eval_date = '2022-01-01'
        end_eval_date = datetime.today().strftime('%Y-%m-%d')
        dataset_total = yf.download(ticker, start='2010-01-01', end=end_eval_date, progress=False)
        real_stock_price = dataset_total[dataset_total.index >= start_eval_date][['Open']].values

        inputs = dataset_total[len(dataset_total) - len(real_stock_price) - 60:][['Open']].values
        inputs_scaled = loaded_scaler.transform(inputs)
        
        X_test = []
        for i in range(60, len(inputs_scaled)):
            X_test.append(inputs_scaled[i-60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        if len(real_stock_price) != len(X_test):
             min_len = min(len(real_stock_price), len(X_test))
             logging.warning(f"Ajustando dimensiones de {len(real_stock_price)} y {len(X_test)} a {min_len}")
             real_stock_price = real_stock_price[:min_len]
             X_test = X_test[:min_len]

        predicted_stock_price_scaled = loaded_model.predict(X_test)
        predicted_stock_price = loaded_scaler.inverse_transform(predicted_stock_price_scaled)
        
        mse = mean_squared_error(real_stock_price, predicted_stock_price)
        mae = mean_absolute_error(real_stock_price, predicted_stock_price)
        rmse = np.sqrt(mse)
        r2 = r2_score(real_stock_price, predicted_stock_price)

        logging.info(f"\nMétricas de Evaluación: MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

        mlflow.set_experiment(eval_experiment_name)
        with mlflow.start_run() as eval_run:
            logging.info(f"Registrando métricas de evaluación en MLflow. Run ID: {eval_run.info.run_id}")
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("parent_training_run_id", training_run_id)

        logging.info("--- Evaluación completada. ---")

    ticker_param = "{{ params.ticker }}" 

    training_run_id_value = train_model(ticker=ticker_param, experiment_name=TRAINING_EXPERIMENT_NAME)
    evaluate_model(training_run_id=training_run_id_value, ticker=ticker_param, eval_experiment_name=EVALUATION_EXPERIMENT_NAME)

stock_prediction_pipeline()
