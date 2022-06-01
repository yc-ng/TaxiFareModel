import mlflow
from mlflow.tracking import MlflowClient

# Indicate mlflow to log to remote server
mlflow.set_tracking_uri("https://mlflow.lewagon.ai/")

# [country code] [city] [login] model name + version
EXPERIMENT_NAME = "[SG][Singapore][YC]taxi_fare_predict_v1"

client = MlflowClient()

try:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
except BaseException:
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

yourname = 'YC'

if yourname is None:
    print("please define your name, it will be used as a parameter to log")

for model in ["linear", "Randomforest"]:
    run = client.create_run(experiment_id)
    client.log_metric(run.info.run_id, "rmse", 4.5)
    client.log_param(run.info.run_id, "model", model)
    client.log_param(run.info.run_id, "student_name", yourname)
