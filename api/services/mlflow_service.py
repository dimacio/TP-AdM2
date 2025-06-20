import mlflow.keras
import os
import pickle
from mlflow.tracking import MlflowClient
from schemas import MetricType

def load_model_and_scaler(run_id: str):
    """
    Carga el modelo y el scaler desde MLflow usando el run_id.
    :param run_id: ID del run de MLflow.
    :return: modelo y scaler cargados.
    """
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.keras.load_model(model_uri)

    client = MlflowClient()
    local_path = client.download_artifacts(run_id, "scaler_artifact")
    scaler_path = os.path.join(local_path, "scaler.pkl")
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def get_best_run_id(experiment_name: str, metric: MetricType) -> str:
    """
    Busca el parent_training_run_id del mejor modelo según la métrica especificada.
    :param experiment_name: Nombre del experimento en MLflow.
    :param metric: Nombre de la métrica (por ejemplo, 'rmse', 'r2').
    :param ascending: True para minimizar la métrica (ej: rmse), False para maximizar (ej: r2).
    :return: parent_training_run_id del mejor modelo.
    """ 

    ascending = metric in ["rmse", "mae", "mse"]

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1
    )
    if not runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")
    parent_run_id = runs[0].data.params.get("parent_training_run_id")
    if not parent_run_id:
        raise ValueError(f"parent_training_run_id not found in best run for experiment '{experiment_name}'.")
    return parent_run_id