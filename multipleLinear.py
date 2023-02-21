import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("C:/Users/ryan_/Documents/Programming/repos/MLCourse/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 5 - Multiple Linear Regression/Python/50_startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, :-1].values

#Encoding categorical into usable data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
