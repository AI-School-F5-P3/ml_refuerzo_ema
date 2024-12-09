import mlflow
import mlflow.tensorflow
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def setup_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000")  # URL del servidor de MLflow
    mlflow.set_experiment("Customer Classification")

def log_metrics_and_model(metrics, model_path):
    setup_mlflow()
    with mlflow.start_run():
        # Loguear métricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Loguear el modelo
        mlflow.tensorflow.log_model(tf_saved_model_dir=model_path, artifact_path="model")
        print(f"Modelo y métricas logueados en MLflow.")
