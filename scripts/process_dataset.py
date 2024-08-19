import os

from sklearn.model_selection import train_test_split
import pandas as pd

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))


def split_data(**context):
    data = pd.read_csv('./data/Iris.csv')
    data = data.drop(columns=['Id'])

    X = data.drop(columns=['Species'])
    Y = data['Species']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

    x_train.to_csv(f'{root_dir}/data/x_train.csv', index=False)
    x_test.to_csv(f'{root_dir}/data/x_test.csv', index=False)
    y_train.to_csv(f'{root_dir}/data/y_train.csv', index=False)
    y_test.to_csv(f'{root_dir}/data/y_test.csv', index=False)

    context['ti'].xcom_push(key='root_dir', value=root_dir)

