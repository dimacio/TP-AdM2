# Operaciones de Aprendizaje de Máquina I (CEIA-MLOps1)

**Profesores:**  
Facundo Adrián Lucianna - facundolucianna@gmail.com

## Descripción del Proyecto

Este proyecto implementa un pipeline de machine learning para predicción de precios de acciones utilizando datos históricos descargados de Yahoo Finance. El flujo completo está orquestado con Airflow y utiliza MLflow para el tracking de experimentos y MinIO como almacenamiento de artefactos. El modelo principal es una red LSTM entrenada para predecir el precio de apertura de una acción (por defecto NVDA). Además, se provee una API (FastAPI) para exponer el modelo entrenado y realizar predicciones en tiempo real.

## Integrantes

- a1721 Dimas Ignacio Torres (dimaciodimacio@gmail.com)
- a1726 Joaquín Matías Mestanza (joa.mestanza@gmail.com)
- a1714 Ramiro Andrés Feichubuinm (ra.feichu@gmail.com)

## Como correr el proyecto

Estando en root correr:

```
docker-compose up --build
```

### Rutas útiles

- MLFlow: http://localhost:5001
- MinIO (Bucket): http://localhost:9001
- Airflow API Docs: http://localhost:8080/api/v1/ui/

### Comandos útiles para entrenamiento

Realizar una nueva run:

```
docker-compose restart yahoo_trainer
```

Para ver log de errores de script:

```
docker-compose logs yahoo_trainer
```

Después de haber hecho un cambio en el script de python:

```
docker-compose build yahoo_trainer
docker-compose up -d yahoo_trainer
```
