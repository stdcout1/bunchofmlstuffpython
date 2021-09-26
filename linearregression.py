import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pp
import pickle
from matplotlib import style

data = pd.read_csv('student-mat.csv', sep = ';')

data = data[['G1', 'G2', 'G3', 'goout', 'failures', 'absences']]
print(data.head())
predict = 'G3'
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train ,y_test = sklearn.model_selection.train_test_split(x,y, test_size= 0.1)

line = linear_model.LinearRegression()

line.fit(x_train,y_train)
acc = line.score(x_test,y_test)
print(acc)

pre = line.predict(x_test)

with open('studentmode.pickle', 'wb') as f:
    pickle.dump(line,f)
pickle_in = open('studentmode.pickle')

for i in range(len(pre)):
    print(pre[x],x_test[x],y_test[x])