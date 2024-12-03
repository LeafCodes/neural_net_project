import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('/Users/ASUS/Downloads/data')
data = pd.read_csv('datalemari.csv', sep=',', header=0)
data.head()

d = data.iloc[:,3].values
x = np.transpose(data.iloc[:,:3].values)
w = np.random.rand(3).reshape(1,-1)
l_r = 0.1

def func (w, x):
    y = (-w[0,0]/w[0,2])+(-w[0,1]/w[0,2])*x[1,:]
    return y

def error_J (Jb, Jl):
    e = np.abs((Jb - Jl)/Jb)
    return e

def Jf (d, y) :
    e = d-y
    J = np.sum(e*e)
    return J

def plot (x, y):
    plt.plot(x, y)
    plt.show()

def dJi (i, x, y, d):
    dJ = np.sum(2*(d-y)*x[i,:])
    return dJ

def d2J (i, x):
    d2J = np.sum(2*(x[i,:])*(x[i,:])*-1)
    return d2J

y = np.dot(w,x)
J = Jf(d, y)
d2J_dw0 = d2J(0, x)
d2J_dw1 = d2J(1, x)
d2J_dw2 = d2J(2, x)

plt.plot(x[1,0:5], x[2,0:5], 'or')
plt.plot(x[1,6:12], x[2,6:12], 'og')
plt.plot(x[1,13:16], x[2,13:16], 'ob')
plt.xlabel('Lebar')
plt.ylabel('Tinggi')

def nr(i, w, x, l_r_t, d2Ji, galat, iter):
    for iter_t in range(iter):
        y = np.dot(w,x)
        dJ = dJi(i, x, y, d)
        wbi = w[0,i] - ((l_r_t/d2Ji)*dJ)
        if np.abs(dJ) <= galat :
            break
        w[0,i] = wbi
    return w

for k in range (100):
    w = nr(0, w, x, l_r, d2J_dw0, 0.01, 100)
    w = nr(1, w, x, l_r, d2J_dw1, 0.01, 100)
    w = nr(2, w, x, l_r, d2J_dw2, 0.01, 100)

    y = np.dot(w,x)
    Jb = Jf(d,y)
    e = error_J(Jb, J)
    if e <= 0.0001:
        break
    J = Jb

        
garis = func(w,x)
plot(x[1,:],garis)
print(J)