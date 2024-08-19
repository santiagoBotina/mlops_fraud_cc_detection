import os
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

# Make available the tasks in the scripts directory
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
sys.path.insert(0, scripts_dir)

from process_dataset import split_data
from model_training import train_model

dag = DAG(
    default_args={
        "depends_on_past": False,
        "email": ["airflow@example.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 0,
    },
    schedule=None,
    dag_id="refactored_iris_model_training_dag",
    description="DAG for iris model training",
    tags=["iris_model"]
)

process_data_task = PythonOperator(
    task_id="split_data",
    python_callable=split_data,
    dag=dag
)

train_model_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag
)

process_data_task >> train_model_task
