import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import pickle
import mlflow


#def train():
x_train = pd.read_csv('data/x_train.csv')
x_test = pd.read_csv('data/x_test.csv')
y_train = pd.read_csv('data/y_train.csv')
y_test = pd.read_csv('data/y_test.csv')

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

print("Accuracy: ", model.score(x_test, y_test) * 100)

y_pred = model.predict(x_test)
print("Accuracy Score : ", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

# MLflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("MLflow Quickstart")

with open('models/iris_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with mlflow.start_run():
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
