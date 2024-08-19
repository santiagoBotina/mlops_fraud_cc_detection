# Iris Detection MLOps Project - by Santiago Botina
Welcome to the Iris Detection MLOps project! This project is a simple example of how to deploy a machine learning model using MLOps practices. \
The project uses the Iris dataset to train a simple decision tree classifier and deploy it using Flask.
# Installation

## Create virtual ENV and activate it

```bash
python -m venv venv && source venv/bin/activate
```

## Dependencies

```bash
pip install -r requirements.txt
```

## Configure resources
### Airflow and MLflow

1. Create the following variables in Airflow:
```bash
export AIRFLOW_HOME=$(pwd)/airflow
export AIRFLOW_VERSION=2.10.0
export PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
```

2. Create user `admin` with password `admin` as well:
```bash
airflow users create \
    --username admin \
    --firstname admin \
    --lastname admin \
    --role Admin \
    --email admin \
    --password admin
    
```

3. Now run the following command to check if the variables are set correctly:
```bash
airflow standalone
```
* Now you can access the airflow web server by going to `localhost:8080` in your browser

4. If the command runs without any errors, you can now run mlflow ui server to track experiments:
* Note that the value inside `--backend-store-uri` should be the FULL path to the mlflow directory in the project \
    (e.g. `/Users/user/Desktop/personal/mlops/mlops_iris_detection/mlflow`)
```bash
mlflow ui --backend-store-uri YOUR/PATH/TO/mlflow
```

5. Now you can access the mlflow ui server by going to `localhost:5000` in your browser.