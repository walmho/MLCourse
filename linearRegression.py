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

fig, (train, test) = plt.subplots(2)
#Training results
train.set_title('Salary vs. Experience (Training)')
train.set_xlabel('Years of Experience')
train.set_ylabel('Salary')

train.scatter(X_test, Y_test, color='blue')
train.plot(X_train, regressor.predict(X_train))

#Test results
test.set_title('Salary vs. Experience')
test.set_xlabel('Years of Experience')
test.set_ylabel('Salary')

plt.show()
