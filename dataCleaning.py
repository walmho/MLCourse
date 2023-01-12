#Import Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Import Dataset
dataset = pd.read_csv("Machine Learning A-Z (Codes and Datasets)/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Python/Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, -1].values
#print(f"x: {X}, y: {Y}")

#Taking care of missing dataset
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
ct.fit_transform(X)
X = np.array(ct.fit_transform(X))

#Encoding the Dependnent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

#Splitting data into test and training set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=1)
#print(f"{X_train}\n{X_test}\n{Y_train}\n{Y_test}")

#Feature Scaling (Note: Doesn't need to be implemented for some ML models)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(f"{X_train}, {X_test}")
