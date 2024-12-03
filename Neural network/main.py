import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import pandas as pd

os.chdir('/Users/ASUS/Downloads/data')
data = pd.read_csv('data_lemari.csv', sep=',', header=0)
data.head()

y = data.iloc[:,2]
y2 = data.iloc[:,3]
y3 = data.iloc[:,4]
X = data.iloc[:,:2]
l = data.iloc[:,0]
t = np.empty([12])
t2 = np.empty([12])
t3 = np.empty([12])

model = LinearRegression().fit(X, y)
r_sq = model.score(X,y)
#print(f"coefficient of determination: {r_sq}")
#print(f"intercept: {model.intercept_}")
#print(f"slope: {model.coef_}")

Pred = model.predict(X.iloc[:4,:])

def func(x, w1, w2, c):
    y = (w1/w2)*x + (1/w2)*c
    
    return y
print(f"{r_sq}")

for i in range (12):
    t[i] = func(l[i], model.coef_[0], -1*model.coef_[1], model.intercept_)

llt = LinearRegression().fit(X,y2)

for i in range (12):
    t2[i] = func(l[i], llt.coef_[0], -1*llt.coef_[1], llt.intercept_)

ltw = LinearRegression().fit(X,y3)

for i in range (12):
    t3[i] = func(l[i], ltw.coef_[0], -1*ltw.coef_[1], ltw.intercept_)

plt.plot(X.iloc[:,0], X.iloc[:,1], 'o')
plt.plot(l, t,'-r')
plt.plot(l, t2,'-g')
plt.plot(l, t3,'-b')
plt.xlabel('Lebar')
plt.ylabel('Tinggi')
plt.show()