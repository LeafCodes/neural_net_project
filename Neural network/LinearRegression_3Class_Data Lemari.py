import numpy as np
from sklearn.linear_model import LinearRegression
import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('/Users/ASUS/Downloads/data')
data = pd.read_csv('datalemari(-1,1).csv', sep=',', header=0)
data.head()

d1 = data.iloc[:,2].values
d2 = data.iloc[:,3].values
d3 = data.iloc[:,4].values
x1 = data.iloc[:,0:2].values
x2 = data.iloc[:,0].values

model1 = LinearRegression().fit(x1,d1)
r_sq1 = model1.score(x1,d1)
model2 = LinearRegression().fit(x1,d2)
r_sq2 = model2.score(x1,d2)
model3 = LinearRegression().fit(x1,d3)
r_sq3 = model3.score(x1,d3)

def func(x, w1, w2, c):
    y = (w1/w2)*x + (1/w2)*c
    return y

sumbu_x = np.array([[0.5], [2.5]])
y1 = func(sumbu_x,  model1.coef_[0], -1*model1.coef_[1], model1.intercept_)
y2 = func(sumbu_x,  model2.coef_[0], -1*model2.coef_[1], model2.intercept_)
y3 = func(sumbu_x,  model3.coef_[0], -1*model3.coef_[1], model3.intercept_)

y_pred1 = model1.predict(x1).reshape(16,1)
y_pred2 = model2.predict(x1).reshape(16,1)
y_pred3 = model3.predict(x1).reshape(16,1)

plt.plot(x1[0:5,0], x1[0:5,1], 'or', label = 'lemari')
plt.plot(x1[5:12,0], x1[5:12,1], 'ob', label = 'buffet')
plt.plot(x1[12:16,0], x1[12:16,1], 'og', label = 'wardrobe')
plt.plot(sumbu_x, y1, '-r')
plt.plot(sumbu_x, y2, '-b')
plt.plot(sumbu_x, y3, '-g')
plt.xlabel('Lebar')
plt.ylabel('Tinggi')
plt.xlim(0.5, 2.5)
plt.ylim(0, 2.5)    
plt.title('Regresi Linear')
plt.legend()

datas = np.append(y_pred1, y_pred2, axis=-1)
datas = np.append(datas, y_pred3, axis=-1)
print('Data : \n', x1)
print('Hasil : \n', datas)

plt.show()