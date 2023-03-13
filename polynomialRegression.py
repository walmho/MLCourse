# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:16:59 2023

@author: ryan_
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("C:/Users/ryan_/Documents/Programming/repos/MLCourse/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

lin_reg = LinearRegression()
lin_reg.fit(x, y)

degree_number = 6
poly_reg = PolynomialFeatures(degree = degree_number)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

### LINEAR ONLY
# plt.scatter(x, y, color='red')
# plt.plot(x, lin_reg.predict(x), color='blue')
# plt.title("Truth or Bluff (linear regression version)")
# plt.xlabel("Position Level")
# plt.ylabel("Salary")
# plt.show()

### POLYNOMIAL ONLY
# plt.scatter(x, y, color='red')
# plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color='purple')
# plt.title("Truth or Bluff (polynomial regression version)")
# plt.xlabel("Position Level")
# plt.ylabel("Salary")
# plt.show()

### COMPARISON VERSION
# plt.scatter(x, y, color='red', label='current salary points')
# plt.plot(x, lin_reg.predict(x), color='blue', label='predicted linear')
# plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color='purple', label=f'polynomial, degree {degree_number}')
# plt.title("Truth or Bluff (regression comparison version)")
# plt.xlabel("Position Level")
# plt.ylabel("Salary")
# plt.legend(loc='upper left')
# plt.show()

### Smoother, high res curve
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red', label='current salary points')
plt.plot(x_grid, lin_reg.predict(x_grid), color='blue', label='predicted linear')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color='purple', label=f'polynomial, degree {degree_number}')
plt.title("Truth or Bluff (regression comparison version)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.legend(loc='upper left')
plt.show()
