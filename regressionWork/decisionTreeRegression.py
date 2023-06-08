import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

#Importing data
dataset = pd.read_csv('C:/Users/ryan_/Documents/Programming/repos/MLCourse/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 8 - Decision Tree Regression/Python/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Training on whole dataset
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)
regressor.predict([[6.5]])

#Visualize results
# DTR IS NOT WELL SUITED FOR 2D RESULTS
x_grid = np.arange(min(X), max(X), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Salary vs. Experience (Decision Tree Regression version)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()