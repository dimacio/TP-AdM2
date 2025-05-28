# Dockerfile para el script de entrenamiento yahoo_trainer

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el script de entrenamiento y cualquier otro archivo necesario por el script
COPY train_mlflow.py .
# Si tuvieras un archivo plots.py y lo usaras, lo copiarías también:
# COPY plots.py . 

# Directorio para datos (el volumen se mapeará aquí desde docker-compose)
RUN mkdir data
# Directorio para gráficos (el volumen se mapeará aquí desde docker-compose)
RUN mkdir plots

# Comando para ejecutar el script
CMD ["python", "train_mlflow.py"]
