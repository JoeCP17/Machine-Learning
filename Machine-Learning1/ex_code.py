import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('SLR2.csv')

X = dataset['fertility rate']
y = dataset['worker percent']

X = np.array(X).reshape(-1, 1)
y = np.array(y)

from sklearn.model_selection import train_test_split  # test , train 각각 나누어 학습

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# X와 y의 각 train , test 나누어 학습

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)
print(X_test_std)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train_std, y_train)
print(regressor.score(X_train_std, y_train))
y_pred = regressor.predict(X_test_std)

from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# plt를 통해 그래프 수치화
plt.scatter(X_train_std, y_train, color='red')
plt.plot(X_train_std, regressor.predict(X_train_std), color='blue')
plt.title('Participation in Workforce vs Fertility Rate')
plt.xlabel('$x_1$' , fontsize=18)
plt.ylabel('$y$' , rotation = 0 , fontsize=18)

plt.show()
