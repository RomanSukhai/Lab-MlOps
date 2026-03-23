from datetime import datetime
import json
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator


PROJECT_DIR = "/opt/airflow/project"
METRICS_PATH = os.path.join(PROJECT_DIR, "metrics.json")


def choose_branch():
    if not os.path.exists(METRICS_PATH):
        return "skip_registration"

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    f1 = float(metrics.get("test_f1", 0.0))

    if f1 >= 0.50:
        return "register_model"

    return "skip_registration"


with DAG(
    dag_id="ml_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "dvc", "mlflow"],
) as dag:

    check_data = BashOperator(
        task_id="check_data",
        bash_command=f"cd {PROJECT_DIR} && dvc pull || true",
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=f"cd {PROJECT_DIR} && dvc repro prepare",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd {PROJECT_DIR} && dvc repro train",
    )

    optimize_model = BashOperator(
        task_id="optimize_model",
        bash_command=f"cd {PROJECT_DIR} && dvc repro optimize",
    )

    branch_decision = BranchPythonOperator(
        task_id="branch_decision",
        python_callable=choose_branch,
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python -c \"import mlflow; "
            "mlflow.set_tracking_uri('file:./mlruns'); "
            "mlflow.set_experiment('lab-mlops'); "
            "run = mlflow.start_run(); "
            "mlflow.log_artifact('data/models/model.pkl'); "
            "mlflow.log_artifact('data/models/confusion_matrix.png'); "
            "mlflow.log_artifact('metrics.json'); "
            "mlflow.end_run()\""
        ),
    )

    skip_registration = EmptyOperator(
        task_id="skip_registration",
    )

    end = EmptyOperator(
        task_id="end",
    )

    check_data >> prepare_data >> train_model >> optimize_model >> branch_decision
    branch_decision >> register_model >> end
    branch_decision >> skip_registration >> end