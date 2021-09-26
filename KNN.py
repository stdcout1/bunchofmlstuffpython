import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv('abalone.data')
le = preprocessing.LabelEncoder()
sex = le.fit_transform(list(data["Sex"]))
data = data.drop(['Sex'], 1)

predict = 'Rings'
x = np.array(data.drop(predict, 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)