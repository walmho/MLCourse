import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv("C:/Users/ryan_/Documents/Programming/repos/MLCourse/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Python/Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(f"X: {x}\n Y: {y}")

y = y.reshape(len(y), 1)
print(y)

scX = StandardScaler()
scY = StandardScaler()
X = scX.fit_transform(x)
Y = scY.fit_transform(y)
print(X)
print(Y)

regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

scY.inverse_transform(regressor.predict(scX.transform([[6.5]])).reshape(-1,1))

plt.scatter(scX.inverse_transform(X), scY.inverse_transform(Y), color='red')
plt.plot(scX.inverse_transform(X), scY.inverse_transform(regressor.predict(X).reshape(-1,1)))
plt.title("Truth or Bluff SVR")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
