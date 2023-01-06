#Import Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Import Dataset
dataset = pd.read_csv("Machine Learning A-Z (Codes and Datasets)/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Python/Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, -1].values
print(f"x: {X}, y: {Y}")

#Taking care of missing dataset
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
