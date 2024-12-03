import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer

#Deklarasi Data
os.chdir('/Users/ASUS/Downloads/data')
data = pd.read_csv('datalemari(-1,1).csv', sep=',', header=0)
data.head()

d = data.iloc[:,2:5].values            
x = data.iloc[:,:2].values
N = x.shape[0]
x = np.append(np.ones((N, 1)), x, axis=-1)
i_total = x.shape[1]
j_total = d.shape[1]
w = np.ones((i_total, j_total))                
alpha = 0.1     
print('x :\n', x)
print('w :\n', w)                                        

def func (w, x):
    N = x.shape[0]
    y = - ((w[1,0]*x/w[2,0])+(w[0,0]/w[2,0])).reshape(N,1)
    for i in range (1, w.shape[1]):                                    
        y1 = - ((w[1,i]*x/w[2,i])+(w[0,i]/w[2,i])).reshape(N,1)
        y = np.append(y, y1, axis=-1)
    return y

def forward (x,w) :
    y = np.dot(x,w)
    return y

def J (d, y):
    output = np.sum((d-y)**2)
    return output

def dJ_dw (x, d, y_pred):
    output = np.dot(2*x.T, (d-y_pred))
    return output

def D2J_dw (x):
    output = np.sum(-2 * x**2, axis=0)
    return output

def train (x, w, d, i_, j_, alpha, iterasi, galat):
    alpha = alpha/D2J_dw(x)
    for i in range (i_):
        for j in range (j_):    
            for k in range (iterasi):
                y_pred = forward(x,w)
                dj_dw = dJ_dw(x, d, y_pred)
                w_t = w[i, j] - (alpha[i]*dj_dw[i, j])
                if galat > np.abs(w_t-w[i, j]):
                    break
                w[i, j] =w_t
    return w
error = np.zeros(1000)
Start = timer()
for epoch in range (1000):
    y_pred = forward(x, w)
    w = train(x, w ,d , i_total, j_total, alpha, 100, 0.001)
    error[epoch] = J(d, y_pred)
end = timer()
print("Waktu Komputasi :",end-Start)
print('y prediksi :\n', forward(x, w))
print('d :\n', d)

sumbu_x = np.array([[0], [2.5]])
garis = func(w, sumbu_x)

print('weight :\n', w)

plt.plot(x[0:5,1], x[0:5,2], 'or', label='Lemari')
plt.plot(x[5:12,1], x[5:12,2], 'ob', label='Buffet')
plt.plot(x[12:16,1], x[12:16,2], 'og', label='wardrobe')
plt.plot(sumbu_x, garis[:,0], '-r')
plt.plot(sumbu_x, garis[:,1], '-b')
plt.plot(sumbu_x, garis[:,2], '-g')
plt.xlabel('Lebar')
plt.ylabel('Tinggi')
plt.xlim(0.5, 2.5)
plt.ylim(0, 2.5)
plt.title('Neural Network')
plt.legend()
plt.show()
