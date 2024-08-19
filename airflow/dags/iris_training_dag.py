import os

from airflow.decorators import dag, task
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.model_selection import train_test_split
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import pickle
import mlflow

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
def process_data():
    data = pd.read_csv('./data/Iris.csv')
    data = data.drop(columns=['Id'])

    X = data.drop(columns=['Species'])
    Y = data['Species']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

    x_train.to_csv(f'{data_dir}/x_train.csv', index=False)
    x_test.to_csv(f'{data_dir}/x_test.csv', index=False)
    y_train.to_csv(f'{data_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{data_dir}/y_test.csv', index=False)


def train_model():
    x_train = pd.read_csv(f'{data_dir}/x_train.csv')
    x_test = pd.read_csv(f'{data_dir}/x_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv')
    y_test = pd.read_csv(f'{data_dir}/y_test.csv')

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    print("Accuracy: ", model.score(x_test, y_test) * 100)

    y_pred = model.predict(x_test)
    print("Accuracy Score : ", accuracy_score(y_test, y_pred))

    print(classification_report(y_test, y_pred))

    with open(f'{model_dir}/iris_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # MLflow
    #mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment("Iris model training")

    mlflow.start_run()
    mlflow.log_param("params", model.get_params())
    mlflow.log_metric("accuracy", model.score(x_test, y_test) * 100)
    mlflow.log_metric("accuracy_score", accuracy_score(y_test, y_pred))

    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    signature = infer_signature(x_train, model.predict(x_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="iris_model",
        signature=signature,
        input_example=x_train,
        registered_model_name="tracking-quickstart",
    )

    # Load the model back for predictions as a generic Python Function model
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    predictions = loaded_model.predict(x_test)

    iris_feature_names = datasets.load_iris().feature_names

    result = pd.DataFrame(x_test, columns=iris_feature_names)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions

    result.to_csv(f'{data_dir}/result.csv', index=False)

    mlflow.log_artifact(f'{data_dir}/result.csv')
    mlflow.end_run()


dag = DAG(
    default_args={
        "depends_on_past": False,
        "email": ["airflow@example.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 0,
    },
    schedule=None,
    dag_id="iris_model_training_dag",
    description="DAG for iris model training",
    tags=["iris_model"]
)

process_data_op = PythonOperator(
    task_id="process_data",
    python_callable=process_data,
    dag=dag,
)

train_op = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,
)

process_data_op >> train_op
