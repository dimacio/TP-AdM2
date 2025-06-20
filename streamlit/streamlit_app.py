import streamlit as st
import requests
import os
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta

API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# st.title("Predicción de apertura NVDA")

# Predicción automática al cargar la página
# @st.cache_data
# def get_prediction():
#     try:
#         response = requests.post(f"{API_URL}/predict")
#         response.raise_for_status()
#         pred = response.json()["prediction"]
#         return pred
#     except Exception as e:
#         return f"Error: {e}"

# pred = get_prediction()
# if isinstance(pred, float):
#     st.success(f"La predicción de apertura es: ${pred:.2f}")
# else:
#     st.error(pred)

# # Descargar y graficar últimos 60 días de NVDA
# @st.cache_data
def get_nvda_data():
    ticker_symbol = "NVDA"
    ticker = yf.Ticker(ticker_symbol)

    historical_data = ticker.history(period="60d")

    filtered_data = historical_data[['Open']]
    return filtered_data


df = get_nvda_data()
print(df.columns)
# st.subheader("Precio de apertura (Open) de NVDA - Últimos 60 días")
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=df.index, y=df["Open"], mode="lines+markers", name="Apertura"))
# fig.update_layout(xaxis_title="Fecha", yaxis_title="Precio de apertura (USD)")
# st.plotly_chart(fig, use_container_width=True)
