import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle

data = pd.read_csv('TDY.csv')
print(data.head())
data = data[['Open', 'Close']]
print(data.head())

predict = "Close"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train ,y_test = sklearn.model_selection.train_test_split(x,y, test_size= 0.1)

line = linear_model.LinearRegression()

line.fit(x_train,y_train)
acc = line.score(x_test,y_test)
print(acc)
d = {'Open': [419.98]}
df = pd.DataFrame(data = d)
pre = line.predict(df)
print(pre)