# Stock Prediction API

API REST construida con **FastAPI** para exponer modelos de predicción de precios de apertura de acciones, autenticación vía Airflow y gestión de experimentos con MLflow.

## 🚀 Cómo ejecutar la API

Desde este directorio (`/api`):

```sh
docker build -t stock-predictor-api .
docker run -p 8000:8000 stock-predictor-api
```

O usando Docker Compose (recomendado desde el root del proyecto):

```sh
docker-compose up --build api
```

Accede a la documentación interactiva en:  
[http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📂 Estructura de carpetas

```
api/
├── main.py                # FastAPI app, endpoints y dependencias
├── schemas.py             # Modelos Pydantic para validación de datos
├── services/
│   ├── mlflow_service.py  # Lógica de MLflow: carga de modelos y selección de runs
│   ├── auth_service.py    # Lógica de autenticación JWT y validación con Airflow
│   └── __init__.py
├── requirements.txt
├── README.md
└── __init__.py
```

---

## 🛠️ Endpoints principales

- **POST `/login`**  
  Autenticación contra Airflow. Devuelve un JWT si las credenciales son válidas.

- **POST `/trigger-new-dag-run`**  
  Lanza una nueva ejecución de un DAG de Airflow (requiere JWT).

- **GET `/get-best-model-run-id`**  
  Obtiene el `run_id` del mejor modelo registrado en MLflow según la métrica seleccionada.

- **POST `/predict-sample`**  
  Realiza una predicción de precio de apertura para un ticker y fecha dados usando el mejor modelo disponible.

- **GET `/`**  
  Health check.

---

## 🔒 Autenticación

- El endpoint `/login` valida usuario y contraseña contra la API de Airflow.
- El JWT devuelto debe enviarse en el header `token` para endpoints protegidos.
- Las contraseñas se almacenan temporalmente en Redis para interactuar con Airflow.

---

## ⚙️ Dependencias principales

- FastAPI, Uvicorn
- MLflow
- Redis
- yfinance
- TensorFlow/Keras
- PyJWT

Instala dependencias para desarrollo local:

```sh
pip install -r requirements.txt
```

---

## 📄 Ejemplo de uso

### Login

```sh
curl -X POST "http://localhost:8000/login" -d "username=admin&password=admin"
```

### Predicción

```sh
curl -X POST "http://localhost:8000/predict-sample" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "NVDA", "date": "2024-06-01", "metric": "rmse"}'
```

---

## 📝 Notas

- La API depende de servicios externos: Airflow, MLflow y Redis (ver [`docker-compose.yml`](../docker-compose.yml)).
- El modelo y el scaler se descargan dinámicamente desde MLflow usando el mejor `run_id` según la métrica seleccionada.
- Para más detalles sobre la arquitectura general, consulta el [README principal](../README.md).
