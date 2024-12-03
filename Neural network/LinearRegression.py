import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import pandas as pd

os.chdir('/Users/ASUS/Downloads/data')
data = pd.read_csv('data_lemari.csv', sep=',', header=0)
data.head()

y = data.iloc[:,2]
X = data.iloc[:,:2]

model = LinearRegression().fit(X, y)
r_sq = model.score(X,y)
#print(f"coefficient of determination: {r_sq}")
#print(f"intercept: {model.intercept_}")
#print(f"slope: {model.coef_}")

Pred = model.predict(X.iloc[4:,:])
#print(f'{Pred}')