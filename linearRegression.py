import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

dataset = pd.read_csv("C:/Users/ryan_/Documents/Programming/repos/MLCourse/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=1)