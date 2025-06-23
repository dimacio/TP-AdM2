import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import pickle
import os
import logging

def get_training_run_id_from_eval_run(eval_run_id: str) -> str:
    """
    Obtiene el ID del run de entrenamiento "padre" a partir de un run de evaluación.
    """
    client = MlflowClient()
    try:
        run_data = client.get_run(eval_run_id).data
        parent_run_id = run_data.params.get("parent_training_run_id")
        if not parent_run_id:
            raise ValueError(f"El Run ID '{eval_run_id}' no tiene el parámetro 'parent_training_run_id'.")
        return parent_run_id
    except MlflowException as e:
        raise ValueError(f"No se pudo obtener el run '{eval_run_id}' desde MLflow: {e}")


def load_model_and_scaler(run_id: str):
    """
    Carga un modelo Keras y un scaler desde un run_id de entrenamiento específico.
    """
    client = MlflowClient()
    logging.info(f"Cargando artefactos desde el Run ID de entrenamiento: {run_id}")
    
    # 1. Cargar el modelo
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.keras.load_model(model_uri)
    logging.info("Modelo cargado exitosamente.")
    
    # 2. Cargar el scaler
    # CORRECCIÓN: Usar el nombre de artefacto correcto definido en el DAG.
    local_dir = client.download_artifacts(run_id, "scaler_artifact")
    scaler_path = os.path.join(local_dir, "scaler.pkl")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logging.info("Scaler cargado exitosamente.")

    return model, scaler


def get_best_run_id(experiment_name: str, metric: str = "rmse"):
    """
    Obtiene el run ID del mejor modelo del experimento de EVALUACIÓN.
    """
    client = MlflowClient()
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"El experimento '{experiment_name}' no fue encontrado en MLflow.")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} ASC"]
        )
        
        if not runs:
            raise ValueError(f"No se encontraron runs en el experimento '{experiment_name}'.")

        return runs[0].info.run_id
    except MlflowException as e:
        raise ValueError(f"Ocurrió un error buscando runs en MLflow: {e}")

