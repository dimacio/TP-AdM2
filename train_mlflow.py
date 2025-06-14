# -*- coding: utf-8 -*-
# train_mlflow_local_data.py

# Librerías
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras 
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # Asegúrate de importar Input

# --- Semillas para que esto sea reproducible ---
np.random.seed(42)
tf.random.set_seed(42)

# --- Parámetros que voy a querer cambiar para experimentar ---

# Datos
# DATA_PATH = './data/yahoo_data.xlsx' # Ruta al archivo de datos local
DATA_PATH = os.getenv("TICKER", "AAPL")
START_DATE_YFINANCE = os.getenv("START_DATE", "2010-01-01")

# TARGET_COLUMN = 'Close*'
TARGET_COLUMN = 'Close' # MODIFY THIS: Was 'Close*'
# FEATURES_COLUMNS = ['Open', 'High', 'Low', 'Adj Close**', 'Volume']
# FEATURES_COLUMNS = ['Open', 'High', 'Low', 'Adj Close', 'Volume'] # MODIFY THIS: Was ['Open', 'High', 'Low', 'Adj Close**', 'Volume']
FEATURES_COLUMNS = ['Open', 'High', 'Low', 'Volume']


# Preprocesamiento y división
LOOKBACK_WINDOW = 60
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15

# Modelo LSTM
LSTM_UNITS_1 = 64
DROPOUT_RATE = 0.2
LSTM_UNITS_2 = 32
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'mean_squared_error'

# Entrenamiento
EPOCHS = 50
BATCH_SIZE = 32

# MLflow
# La URI y el nombre del experimento los tomo de variables de entorno seteadas por docker-compose
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001") # Default si no está en ENV
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Prediccion_Acciones_LSTM_Default")
# Lee el nombre del bucket del entorno, debe coincidir con el creado en docker-compose.yml
MLFLOW_S3_BUCKET_NAME = os.getenv("MLFLOW_BUCKET_NAME", "mlflowbucket") # Default si no está en ENV

# Para guardar los gráficos
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Funciones ---

def cargar_y_preprocesar_datos(data_path, features_cols, target_col):
    """
    Carga, preprocesa y escala los datos.
    Devuelve el df original indexado, los datos escalados y los scalers.
    Si no encuentra el archivo, termina la ejecución.
    """
    # try:
    #     data = pd.read_excel(data_path)
    # except FileNotFoundError:
    #     print(f"Error FATAL: No encontré el archivo de datos en: {data_path}")
    #     return None, None, None, None

    # The data_path argument is now interpreted as the ticker symbol.
    # It uses the global START_DATE_YFINANCE for the start date.
    try:
        ticker_symbol = data_path 
        end_date = datetime.today().strftime('%Y-%m-%d')
        print(f"Descargando datos para el ticker '{ticker_symbol}' desde {START_DATE_YFINANCE} hasta {end_date}...")
        data = yf.download(ticker_symbol, start=START_DATE_YFINANCE, end=end_date, progress=False)

        # --- BEGIN ADDED DEBUG LINE ---
        if not data.empty:
            print(f"DEBUG: Columnas Yahoo Finance: {data.columns.tolist()}")
        # --- END ADDED DEBUG LINE ---

        if data.empty:
            print(f"Error FATAL: No se pudieron descargar datos para '{ticker_symbol}' o el DataFrame está vacío.")
            return None, None, None, None
        print(f"Datos para '{ticker_symbol}' descargados exitosamente. {len(data)} filas.")
    except Exception as e:
        print(f"Error FATAL al descargar datos de Yahoo Finance para '{data_path}': {e}")
        return None, None, None, None


    # Original lines for date processing (commented out or removed):
    # data['Date'] = pd.to_datetime(data['Date'])
    # data = data.sort_values(by='Date')
    # data.set_index('Date', inplace=True)

    # New handling for yfinance data:
    # yfinance data usually has Date as index. Ensure it's a DatetimeIndex and sorted.
    if not isinstance(data.index, pd.DatetimeIndex):
        # This case might occur if 'Date' is a column instead of the index
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
        else: # If 'Date' is not a column and not the index, data is not as expected
            print(f"Error FATAL: DataFrame para '{data_path}' no tiene un índice de fecha (DatetimeIndex) ni una columna 'Date'.")
            return None, None, None, None
    
    # Ensure data is sorted by index (date)
    data.sort_index(inplace=True)

    all_cols_needed = features_cols + [target_col]
    data_subset = data[all_cols_needed].copy()
    
    if data_subset.isnull().any().any():
        print(f"Valores NaN encontrados. Filas antes de dropna: {len(data_subset)}")
        data_subset.dropna(inplace=True)
        print(f"Filas después de dropna: {len(data_subset)}")


    if data_subset.empty:
        print("Error FATAL: DataFrame vacío después de procesar NaNs o carga fallida.")
        return None, None, None, None

    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(data_subset[features_cols])

    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(data_subset[[target_col]])

    scaled_data = np.concatenate((scaled_features, scaled_target), axis=1)
    return data_subset, scaled_data, scaler_features, scaler_target

def crear_secuencias_temporales(data, lookback_window):
    """
    Crea las secuencias para el LSTM.
    """
    X, y = [], []
    if len(data) <= lookback_window:
        print(f"Error: Datos insuficientes (longitud: {len(data)}) para lookback_window de {lookback_window}.")
        return np.array(X), np.array(y)

    for i in range(lookback_window, len(data)):
        X.append(data[i-lookback_window:i, :]) # Todas las features escaladas
        y.append(data[i, -1]) # Solo la última columna (target escalado)
    return np.array(X), np.array(y)

def dividir_datos_cronologicamente(X_seq, y_seq, train_ratio, validation_ratio):
    """
    Divide cronológicamente: train, validation, test.
    """
    n_total = len(X_seq)
    if n_total == 0:
        print("Error: No hay secuencias para dividir.")
        return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * validation_ratio)
    n_test = n_total - n_train - n_val

    if n_train == 0 or n_val == 0 or n_test <= 0:
        print(f"Aviso: División resulta en conjuntos vacíos o inválidos. Total: {n_total}, Train: {n_train}, Val: {n_val}, Test: {n_test}")
        # Podría devolver arrays vacíos aquí si esto es un error fatal para tu lógica
        # return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))


    X_train, y_train = X_seq[:n_train], y_seq[:n_train]
    X_val, y_val = X_seq[n_train : n_train + n_val], y_seq[n_train : n_train + n_val]
    X_test, y_test = X_seq[n_train + n_val:], y_seq[n_train + n_val:]
    return X_train, y_train, X_val, y_val, X_test, y_test

def construir_modelo_lstm(input_shape_tuple, lstm_units_1, dropout_rate, lstm_units_2):
    """
    Arma el modelo LSTM. Dropout entre las LSTM.
    Usa una Input layer para definir explícitamente la forma de entrada.
    """
    model = Sequential()
    model.add(Input(shape=input_shape_tuple)) # Capa de entrada explícita
    model.add(LSTM(units=lstm_units_1, return_sequences=True)) # No necesita input_shape aquí
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=lstm_units_2, return_sequences=False))
    model.add(Dense(units=1))
    return model

def graficar_predicciones(original_dates, y_verdadero_scaled, y_predicho_scaled, titulo, path_guardado, scaler_target):
    """
    Grafica y guarda las predicciones vs los valores reales (en escala original).
    También lo sube a MLflow como artefacto si hay una ejecución activa.
    """
    if len(y_verdadero_scaled) == 0 or len(y_predicho_scaled) == 0:
        print(f"Aviso: Sin datos para graficar en '{titulo}'.")
        return
    
    # Asegurarse de que y_predicho_scaled tenga la misma forma que y_verdadero_scaled para inverse_transform
    if y_predicho_scaled.ndim == 1:
        y_predicho_scaled = y_predicho_scaled.reshape(-1,1)
    if y_verdadero_scaled.ndim == 1:
        y_verdadero_scaled = y_verdadero_scaled.reshape(-1,1)

    # Ajustar la longitud de original_dates si es necesario
    min_len = min(len(original_dates), len(y_verdadero_scaled), len(y_predicho_scaled))
    if len(original_dates) > min_len:
        original_dates = original_dates[-min_len:] # Tomar las últimas fechas
    if len(y_verdadero_scaled) > min_len:
        y_verdadero_scaled = y_verdadero_scaled[-min_len:]
    if len(y_predicho_scaled) > min_len:
        y_predicho_scaled = y_predicho_scaled[-min_len:]

    if min_len == 0: # Si después del ajuste no hay nada que graficar
        print(f"Error: No hay suficientes datos alineados para graficar '{titulo}'.")
        return

    y_verdadero_original = scaler_target.inverse_transform(y_verdadero_scaled)
    y_predicho_original = scaler_target.inverse_transform(y_predicho_scaled)

    plt.figure(figsize=(14, 7))
    plt.plot(original_dates, y_verdadero_original, label='Valor Real (Original)', color='blue', linewidth=1)
    plt.plot(original_dates, y_predicho_original, label='Predicción (Original)', color='orange', linewidth=1, linestyle='--')
    plt.title(titulo)
    plt.xlabel('Fecha')
    plt.ylabel(f'Precio de {TARGET_COLUMN} (Original)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    try:
        plt.savefig(path_guardado)
        if mlflow.active_run(): 
            mlflow.log_artifact(path_guardado)
    except Exception as e:
        print(f"Error guardando/registrando gráfico '{titulo}': {e}")
    plt.close()

def graficar_perdida(historia, path_guardado):
    """
    Grafica la pérdida de entrenamiento/validación.
    MLflow autologging para Keras usualmente guarda esto automáticamente.
    Esta función es un respaldo o para personalización.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(historia.history['loss'], label='Pérdida Entrenamiento')
    if 'val_loss' in historia.history: 
        plt.plot(historia.history['val_loss'], label='Pérdida Validación')
    plt.title('Pérdida del Modelo Durante Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel(f'Pérdida ({LOSS_FUNCTION})')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(path_guardado)
        if mlflow.active_run(): 
            mlflow.log_artifact(path_guardado)
    except Exception as e:
        print(f"Error guardando/registrando gráfico de pérdida: {e}")
    plt.close()

# --- Flujo Principal ---
if __name__ == "__main__":
    print(f"Usando MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            print(f"Creando experimento MLflow: {MLFLOW_EXPERIMENT_NAME}")
            # Construir la ubicación del artefacto usando el nombre del bucket del entorno
            artifact_location_for_experiment = f"s3://{MLFLOW_S3_BUCKET_NAME}/artifacts/{MLFLOW_EXPERIMENT_NAME}"
            print(f"El artifact_location para el nuevo experimento será: {artifact_location_for_experiment}")
            experiment_id = mlflow.create_experiment(name=MLFLOW_EXPERIMENT_NAME, artifact_location=artifact_location_for_experiment)
        else:
             experiment_id = experiment.experiment_id
             print(f"Usando experimento MLflow existente: {MLFLOW_EXPERIMENT_NAME} (ID: {experiment_id}, Artifacts: {experiment.artifact_location})")
        mlflow.set_experiment(experiment_id=experiment_id)

    except Exception as e:
        print(f"Error Crítico configurando el experimento de MLflow '{MLFLOW_EXPERIMENT_NAME}': {e}")
        exit()

    # Habilitar Keras autologging
    # Esto registrará métricas, parámetros, el modelo y los gráficos de pérdida/entrenamiento.
    mlflow.keras.autolog(
        log_models=True, # Guarda el modelo
        log_model_signatures=True, 
        registered_model_name=f"{MLFLOW_EXPERIMENT_NAME}_KerasAutoLog" # Registra el modelo con este nombre
    )

    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
            
        # Loguear parámetros manualmente (autolog también lo hace, pero esto es para asegurar)
        # La interacción entre TensorFlow y MLflow no es óptima

        params_to_log = {
            "data_path": DATA_PATH, "target_column": TARGET_COLUMN,
            "features_columns": str(FEATURES_COLUMNS), "lookback_window": LOOKBACK_WINDOW,
            "train_ratio": TRAIN_RATIO, "validation_ratio": VALIDATION_RATIO,
            "lstm_units_1": LSTM_UNITS_1, "dropout_rate": DROPOUT_RATE,
            "lstm_units_2": LSTM_UNITS_2, "optimizer": OPTIMIZER,
            "loss_function": LOSS_FUNCTION, "epochs": EPOCHS, "batch_size": BATCH_SIZE
        }
        mlflow.log_params(params_to_log)

        data_df_indexed, scaled_data_full, scaler_features, scaler_target = cargar_y_preprocesar_datos(
            DATA_PATH, FEATURES_COLUMNS, TARGET_COLUMN
        )
        if data_df_indexed is None: 
            mlflow.end_run(status="FAILED") # Finalizar run si falla la carga
            exit()

        X_seq, y_seq = crear_secuencias_temporales(scaled_data_full, LOOKBACK_WINDOW)
        if X_seq.shape[0] == 0: 
            mlflow.end_run(status="FAILED")
            exit()

        X_train, y_train, X_val, y_val, X_test, y_test = dividir_datos_cronologicamente(
            X_seq, y_seq, TRAIN_RATIO, VALIDATION_RATIO
        )
        
        if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
            print("Error: Conjuntos de datos vacíos post-división.")
            mlflow.end_run(status="FAILED")
            exit()
        
        input_shape_lstm_tuple = (X_train.shape[1], X_train.shape[2])
        model = construir_modelo_lstm(input_shape_lstm_tuple, LSTM_UNITS_1, DROPOUT_RATE, LSTM_UNITS_2)
        model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION)
        
        print("Entrenando...")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            verbose=1
            # No se necesita mlflow.keras.MLflowCallback(run=run) aquí debido a autolog()
        )
        
        print("Evaluando en conjunto de prueba...")
        loss_test = model.evaluate(X_test, y_test, verbose=0)
        predictions_test_scaled = model.predict(X_test)
        
        r_squared = r2_score(y_test, predictions_test_scaled)
        mse = mean_squared_error(y_test, predictions_test_scaled)
        rmse = np.sqrt(mse)

        print(f"\nMétricas finales (Test):")
        print(f"  Loss (MSE): {loss_test:.6f}")
        print(f"  R2: {r_squared:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}")

        # MLflow autologging ya debería haber logueado estas métricas.
        # Si se quieren loguear explícitamente con un nombre diferente:
        mlflow.log_metric("final_test_loss_mse_manual", loss_test)
        mlflow.log_metric("final_r_squared_manual", r_squared)
        mlflow.log_metric("final_mse_manual", mse)
        mlflow.log_metric("final_rmse_manual", rmse)

        # MLflow autologging para Keras usualmente guarda el gráfico de pérdida.
        # Si se quiere un gráfico personalizado o asegurar que se guarde:
        graficar_perdida(history, os.path.join(PLOTS_DIR, "loss_plot_custom.png"))

        # Graficar predicciones vs reales
        # Calcular el índice de inicio para las fechas del conjunto de prueba
        # El número de secuencias de entrenamiento es X_train.shape[0]
        # El número de secuencias de validación es X_val.shape[0]
        # Cada secuencia usa LOOKBACK_WINDOW puntos, y la predicción es para el punto siguiente.
        # Entonces, la primera fecha para y_test corresponde al índice (LOOKBACK_WINDOW + X_train.shape[0] + X_val.shape[0]) en el dataframe original.
        start_index_for_test_dates = LOOKBACK_WINDOW + X_train.shape[0] + X_val.shape[0]
        
        if start_index_for_test_dates + len(y_test) <= len(data_df_indexed):
            test_dates_original = data_df_indexed.index[start_index_for_test_dates : start_index_for_test_dates + len(y_test)]
            if len(test_dates_original) == len(y_test): 
                 graficar_predicciones(
                    test_dates_original, 
                    y_test.reshape(-1,1), # y_test ya está escalado
                    predictions_test_scaled.reshape(-1,1), # predictions_test_scaled también está escalado
                    'Predicción vs Real (Test - Escala Original)',
                    os.path.join(PLOTS_DIR, "predictions_vs_actual_test_original_scale.png"),
                    scaler_target # El scaler del target para la inversión
                )
            else:
                print(f"Aviso: Discrepancia de longitud final para gráfico de predicciones. Fechas: {len(test_dates_original)}, y_test: {len(y_test)}")
        else:
            print(f"Aviso: No se pudieron alinear las fechas para el gráfico de predicciones del conjunto de prueba. Índice de inicio calculado: {start_index_for_test_dates}, longitud de y_test: {len(y_test)}, longitud de data_df_indexed: {len(data_df_indexed)}")

        # Graficar el precio de cierre histórico completo
        plt.figure(figsize=(14, 7))
        plt.plot(data_df_indexed.index, data_df_indexed[TARGET_COLUMN], label='Precio Cierre Original', color='green')
        plt.title('Precio Cierre Histórico Completo')
        plt.xlabel('Fecha')
        plt.ylabel(f'Precio {TARGET_COLUMN}')
        plt.legend(); plt.grid(True); plt.xticks(rotation=45); plt.tight_layout()
        closing_price_plot_path = os.path.join(PLOTS_DIR, "historical_closing_price_full.png")
        try:
            plt.savefig(closing_price_plot_path)
            if mlflow.active_run():
                mlflow.log_artifact(closing_price_plot_path)
        except Exception as e:
            print(f"Error guardando/registrando gráfico histórico: {e}")
        plt.close()

        # mlflow.keras.log_model() es manejado por autolog() si log_models=True
        # No es necesario llamarlo explícitamente aquí.
        # Si se necesitara, la sintaxis correcta sería:
        # mlflow.keras.log_model(
        # model=model, # 'model' en lugar de 'keras_model'
        # artifact_path="modelo_lstm_acciones_manual",
        # registered_model_name=f"{MLFLOW_EXPERIMENT_NAME}_Model_Manual"
        # )

        print(f"\nMLflow Run completado. Run ID: {run.info.run_id}")
        print(f"UI de MLflow debería estar en: {MLFLOW_TRACKING_URI} (o la IP/DNS de tu servidor MLflow)")
        print(f"Los artefactos deberían estar en el bucket S3: {MLFLOW_S3_BUCKET_NAME} bajo la ruta del experimento/run.")

