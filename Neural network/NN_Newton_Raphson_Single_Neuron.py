import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('/Users/ASUS/Downloads/data')
data = pd.read_csv('datalemari(-1,1).csv', sep=',', header=0)
data.head()

d = data.iloc[:,2].values
x = np.transpose(data.iloc[:,:2].values)
x = np.append(np.ones((1, x.shape[1])), x, axis=0)
w = np.ones(3).reshape(1,-1)
l_r = 0.1
print(w)

plt.plot(x[0,0:5], x[1,0:5], 'or')
plt.plot(x[0,6:12], x[1,6:12], 'og')
plt.plot(x[0,13:16], x[1,13:16], 'ob')
plt.xlabel('Lebar')
plt.ylabel('Tinggi')

def func (w, x):
    y = (-w[0,0]/w[0,2])+(-w[0,1]/w[0,2])*x[1,:]
    return y

def error_J (jb, jl):
    e = np.abs((jb - jl)/jb)
    return e

def Jf (d, y) :
    e = d-y
    J = np.sum(e*e)
    return J

def plot (x, y):
    plt.plot(x, y)
    plt.show()

def dj (i, x, y, d):
    dj = np.sum(2*(d-y)*x[i,:])
    return dj

def d2j (i, x):
    d2j = np.sum(2*(x[i,:])*(x[i,:])*-1)
    return d2j

y = np.dot(w,x)
J = Jf(d, y)
d2j_dw0 = d2j(0, x)
d2j_dw1 = d2j(1, x)
d2j_dw2 = d2j(2, x)

for k in range (100):
    for i in range (100):
        dj0 = dj(0, x, y, d)
        wb0 = w[0,0] - ((l_r/d2j_dw0)*dj0)
        if np.abs(dj0) <= 0.01 :
            break
        w[0,0] = wb0
        y = np.dot(w,x)
       
    for i in range (100):
        y = np.dot(w,x)
        dj1 = dj(1, x, y, d)
        wb1 = w[0,1] - ((l_r/d2j_dw1)*dj1)
        if np.abs(dj1) <= 0.01 :
            break
        w[0,1] = wb1
        
    for i in range (100):
        y = np.dot(w,x)
        dj2 = dj(2, x, y, d)
        wb2 = w[0,2] - ((l_r/d2j_dw2)*dj2)
        if np.abs(dj2) <= 0.01 :
            break
        w[0,2] = wb2

    Jb = Jf(d,y)
    e = error_J(Jb, J)
    if e <= 0.0001:
        break
    J = Jb
    #print(k)
        
print(np.dot(w, x))

    


