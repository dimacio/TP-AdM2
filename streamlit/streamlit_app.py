import streamlit as st
import requests
import os
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta

API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

st.title("Predicción de apertura de acciones")

# Inicializar historial de predicciones en la sesión
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []

# Selección de ticker y métrica
ticker = st.text_input("Ticker", value="NVDA")
metric = st.selectbox("Métrica para seleccionar modelo", ["rmse", "mse", "r2"])

# Obtener fechas de los últimos 60 días hábiles
@st.cache_data
def get_last_60_dates(ticker):
    df = yf.Ticker(ticker).history(period="60d")
    return [d.strftime("%Y-%m-%d") for d in df.index]

dates = get_last_60_dates(ticker)

# Predicción automática al abrir la página
if "predicted_dates" not in st.session_state:
    st.session_state.predicted_dates = set()

progress = st.progress(0, text="Realizando predicciones...")

for i, d in enumerate(dates):
    if d not in st.session_state.predicted_dates:
        payload = {
            "ticker": ticker,
            "date": d,
            "metric": metric
        }
        try:
            response = requests.post(f"{API_URL}/predict-sample", json=payload)
            response.raise_for_status()
            prediction = response.json()["prediction"]
            st.session_state.pred_history.append({
                "ticker": ticker,
                "date": d,
                "prediction": prediction
            })
            st.session_state.predicted_dates.add(d)
        except Exception as e:
            st.session_state.predicted_dates.add(d)  # Para evitar bucles infinitos
    progress.progress((i + 1) / len(dates), text=f"Progreso: {i+1}/{len(dates)}")

progress.empty()

# Mostrar gráfico de los últimos 60 días + TODAS las predicciones hechas
@st.cache_data
def get_ticker_data(ticker):
    df = yf.Ticker(ticker).history(period="60d")
    return df

df = get_ticker_data(ticker)
if not df.empty:
    st.subheader(f"Precio de apertura (Open) de {ticker} - Últimos 60 días")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Open"], mode="lines+markers", name="Apertura"))
    # Graficar todas las predicciones hechas para este ticker
    for pred in st.session_state.pred_history:
        if pred["ticker"] == ticker:
            fig.add_trace(go.Scatter(
                x=[pred["date"]],
                y=[pred["prediction"]],
                mode="markers",
                marker=dict(color="red", size=12, symbol="star"),
                name=f"Predicción {pred['date']}"
            ))
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Precio de apertura (USD)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No hay datos recientes para graficar.")