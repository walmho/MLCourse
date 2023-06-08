import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("C:/Users/ryan_/Documents/Programming/repos/MLCourse/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=1)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

#Training results
plt.title('Salary vs. Experience (Training)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

plt.scatter(X_train, Y_train, color='blue')
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.show()

#Test results
plt.title('Salary vs. Experience (Test)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

plt.scatter(X_test, Y_test, color='blue')
plt.plot(X_train, regressor.predict(X_train), color='red')

plt.show()
