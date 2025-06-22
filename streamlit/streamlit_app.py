import streamlit as st
import requests
import os
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd

# --- CONFIGURACIÓN ---
API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
st.set_page_config(layout="wide")

# --- ESTADO DE LA SESIÓN ---
if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = None
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []

# --- BARRA LATERAL (SIDEBAR) PARA AUTENTICACIÓN ---
with st.sidebar:
    st.title("Autenticación")
    if st.session_state.jwt_token is None:
        username = st.text_input("Usuario (admin)", key="login_user")
        password = st.text_input("Contraseña (admin)", type="password", key="login_pass")
        if st.button("Login", key="login_button"):
            if not username or not password:
                st.error("Por favor, ingrese usuario y contraseña.")
            else:
                try:
                    response = requests.post(f"{API_URL}/login", data={"username": username, "password": password})
                    response.raise_for_status()
                    st.session_state.jwt_token = response.json()["access_token"]
                    st.rerun()
                except requests.HTTPError:
                    st.error("Error de autenticación: Credenciales incorrectas.")
                except Exception as e:
                    st.error(f"Error de conexión: {e}")
    else:
        st.success("✅ Conectado")
        if st.button("Logout", key="logout_button"):
            st.session_state.jwt_token = None
            st.rerun()

# --- CONTENIDO PRINCIPAL ---
st.title("Predicción de apertura de acciones")

ticker = st.text_input("Ticker", value="NVDA", key="ticker_input")
metric = st.selectbox("Métrica para seleccionar modelo", ["rmse", "mse", "r2"], key="metric_select")

if st.session_state.jwt_token:
    if st.button(f"Entrenar nuevo modelo para {ticker}", key="train_button"):
        headers = {"token": st.session_state.jwt_token}
        payload = {"dag_id": "taskflow_stock_prediction_pipeline", "ticker": ticker}
        try:
            response = requests.post(f"{API_URL}/trigger-new-dag-run", json=payload, headers=headers)
            response.raise_for_status()
            st.success(f"¡Entrenamiento iniciado para {ticker}!")
            st.json(response.json())
        except Exception as e:
            st.error(f"Error al iniciar el entrenamiento: {e}")
else:
    st.warning("Por favor, inicie sesión para entrenar un modelo o ver predicciones.")

# --- LÓGICA DE GRÁFICOS Y PREDICCIONES ---
@st.cache_data
def get_ticker_data(ticker_symbol):
    try:
        df = yf.Ticker(ticker_symbol).history(period="60d")
        return df if not df.empty else None
    except Exception:
        return None

df_history = get_ticker_data(ticker)

if df_history is not None:
    st.subheader(f"Precio de apertura (Open) de {ticker} - Últimos 60 días")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_history.index, y=df_history["Open"], mode="lines+markers", name="Apertura Real"))

    # --- SECCIÓN COMPLETADA ---
    # Solo intentar predecir si el usuario está logueado
    if st.session_state.jwt_token:
        headers = {"token": st.session_state.jwt_token}
        try:
            # Llamar al endpoint de predicción de la API
            response = requests.post(f"{API_URL}/predict", json={"ticker": ticker, "metric": metric}, headers=headers)
            response.raise_for_status()
            
            # Procesar y graficar la predicción
            prediction_data = response.json()
            pred_value = prediction_data.get("prediction")
            
            if pred_value is not None:
                # Crear una fecha para el siguiente día hábil para la predicción
                last_date = df_history.index[-1]
                next_day = last_date + pd.Timedelta(days=1)
                fig.add_trace(go.Scatter(x=[next_day], y=[pred_value], mode='markers', name='Predicción', marker=dict(color='red', size=10, symbol='star')))
                st.metric(label=f"Predicción de Apertura para mañana ({ticker})", value=f"${pred_value:,.2f}")
            else:
                st.warning("La API no devolvió una predicción válida.")

        except requests.HTTPError as e:
            # Mostrar el error que viene de la API si no encuentra un modelo
            detail = e.response.json().get("detail", "Error desconocido")
            st.error(f"No se pudo obtener la predicción: {detail}")
        except Exception as e:
            st.error(f"Error al conectar con el servicio de predicción: {e}")
    # --- FIN DE LA SECCIÓN COMPLETADA ---

    fig.update_layout(xaxis_title="Fecha", yaxis_title="Precio de apertura (USD)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error(f"No se pudieron obtener datos para el ticker: {ticker}. Verifique si el ticker es correcto.")