import streamlit as st
import requests
import os
import logging
from datetime import datetime
from enum import Enum

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# --- Modelos de Datos ---
class MetricType(str, Enum):
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"

# --- Funciones de la App ---

def show_login_page():
    st.title("Stock Predictor Login")
    st.info("Usa las credenciales de Airflow (ej: admin/admin)")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if username and password:
                login_url = f"{API_URL}/login"
                try:
                    with st.spinner("Iniciando sesión..."):
                        response = requests.post(
                            login_url,
                            data={"username": username, "password": password},
                            timeout=10
                        )
                        response.raise_for_status()
                    
                    st.session_state['token'] = response.json()['access_token']
                    st.session_state['logged_in'] = True
                    st.rerun()

                except requests.exceptions.HTTPError as e:
                    st.error(f"Fallo el login: {e.response.status_code} - {e.response.text}")
                except requests.exceptions.RequestException:
                    st.error(f"Error de conexión: No se pudo conectar a la API en {login_url}.")
            else:
                st.warning("Por favor, ingresa usuario y contraseña.")

def show_main_app():
    st.title("Predicción de Apertura de Acciones")

    # --- Sección de Predicción ---
    st.header("1. Obtener una Predicción")
    pred_ticker = st.text_input("Ticker Symbol", "NVDA", key="pred_ticker")
    pred_date = st.date_input("Fecha de Predicción", datetime.now())
    pred_metric = st.selectbox("Métrica para seleccionar modelo", [m.value for m in MetricType], key="pred_metric")

    if st.button("Obtener Predicción de Apertura"):
        predict_url = f"{API_URL}/predict-sample"
        payload = {"ticker": pred_ticker, "date": pred_date.strftime("%Y-%m-%d"), "metric": pred_metric}
        headers = {"Authorization": f"Bearer {st.session_state.get('token')}"}
        
        try:
            with st.spinner("Obteniendo predicción..."):
                response = requests.post(predict_url, json=payload, headers=headers, timeout=30)
            
            # --- CORRECCIÓN CLAVE: Mejor manejo de errores ---
            if response.status_code == 404:
                error_detail = response.json().get("detail", "")
                if "No se encontraron runs para el ticker" in error_detail:
                    st.error(f"No se encontró un modelo entrenado para el ticker '{pred_ticker}'.")
                    st.warning("Por favor, entrena un nuevo modelo para este activo usando la sección 2.")
                else:
                    st.error(f"Error 404: {error_detail}")
            else:
                response.raise_for_status()
                prediction = response.json()
                st.success(f"Precio de apertura predicho para {prediction['ticker']} el {prediction['date']}: ${prediction['prediction']:.2f}")

        except requests.exceptions.HTTPError as e:
            st.error("No se pudo obtener la predicción. El servidor respondió con un error.")
            st.json({"url_llamada": e.request.url, "status_code": e.response.status_code, "respuesta_servidor": e.response.json()})
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")

    # --- Sección de Trigger DAG ---
    st.header("2. Entrenar un Nuevo Modelo")
    dag_ticker = st.text_input("Ticker para nuevo entrenamiento", "NVDA", key="dag_ticker")
    
    if st.button(f"Entrenar nuevo modelo para {dag_ticker}"):
        dag_url = f"{API_URL}/trigger-new-dag-run"
        payload = {"dag_id": "stock_prediction_pipeline", "ticker": dag_ticker}
        headers = {"Authorization": f"Bearer {st.session_state.get('token')}"}

        try:
            with st.spinner(f"Iniciando entrenamiento para {dag_ticker}..."):
                response = requests.post(dag_url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
            st.success("¡Se inició el entrenamiento (DAG run) correctamente! Puedes monitorearlo en la UI de Airflow.")
            st.json(response.json())
        except requests.exceptions.HTTPError as e:
            st.error("Error al iniciar el entrenamiento.")
            st.json({"url_llamada": e.request.url, "status_code": e.response.status_code, "respuesta_servidor": e.response.json()})
        except Exception as e:
            st.error(f"Ocurrió un error al iniciar el entrenamiento: {e}")

# --- Lógica principal ---
if not st.session_state.get('logged_in', False):
    show_login_page()
else:
    show_main_app()
