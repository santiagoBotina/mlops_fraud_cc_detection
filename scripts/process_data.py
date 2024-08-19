from sklearn.model_selection import train_test_split
import pandas as pd


data = pd.read_csv('./data/Iris.csv')
data = data.drop(columns=['Id'])

X = data.drop(columns=['Species'])
Y = data['Species']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

x_train.to_csv('data/x_train.csv', index=False)
x_test.to_csv('data/x_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)
