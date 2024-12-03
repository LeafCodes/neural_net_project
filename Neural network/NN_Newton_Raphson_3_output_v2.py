import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer

#Deklarasi Data
os.chdir('/Users/ASUS/Downloads/skripsi/data')
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
k = 0
k_J = []
e = 0.0001
e_total = []
J_total = []

#Deklarasi Fungsi
def func (w, x):
    N = x.shape[0]
    y = - ((w[1,0]*x/w[2,0])+(w[0,0]/w[2,0])).reshape(N,1)
    for i in range (1, w.shape[1]):                                    
        y1 = - ((w[1,i]*x/w[2,i])+(w[0,i]/w[2,i])).reshape(N,1)
        y = np.append(y, y1, axis=-1)
    return y

#Forward bias
def forward (x,w) :
    y = np.dot(x,w)
    return y

#Fungsi menghitung jarak euclidean dari desired target dan actual target
def J (d, y):
    output = np.sum((d-y)**2)
    return output

#Fungsi Turunan Pertama dari jarak euclidean
def dJ_dw (x, d, y_pred):
    output = np.dot(2*x.T, (d-y_pred))
    return output

#Fungsi Turunan Kedua dari jarak euclidean
def D2J_dw (x):
    output = np.sum(-2 * x**2, axis=0)
    return output

#Fungsi untuk melakukan training bobot
def train (x, w, d, i_, j_, alpha, e, k, e_total):
    alpha = alpha/D2J_dw(x)
    for i in range (i_):
        for j in range (j_):    
            while True:
                k += 1
                y_pred = forward(x,w)
                dj_dw = dJ_dw(x, d, y_pred)
                w_t = w[i, j] - (alpha[i]*dj_dw[i, j])
                e_new = np.abs((w_t-w[i, j])/w_t)
                e_total= np.append(e_total, e_new)
                if e > e_new:
                    w[i, j] =w_t
                    break
                w[i, j] =w_t
    return w, k, e_total

y_pred = forward(x,w)
Nilai_J = J(d, y_pred)

Start = timer()
while True:
    w, k, e_total = train(x, w, d, i_total, j_total, alpha, e, k, e_total)
    y_pred = forward(x,w)
    Nilai_J_baru = J(d, y_pred)
    k_J = np.append(k_J, k)
    if e > np.abs((Nilai_J_baru-Nilai_J)/Nilai_J_baru):
        J_total = np.append(J_total, Nilai_J_baru)
        Nilai_J = Nilai_J_baru
        break
    J_total = np.append(J_total, Nilai_J_baru)
    Nilai_J = Nilai_J_baru
end = timer()


sumbu_x = np.array([[0], [2.5]])
garis = func(w, sumbu_x)

print('Jumlah iterasi :', k)
print('Jarak Euclidean Minimum :', min(J_total))
print('Waktu Training :', end-Start)
'''
plt.plot(k_J, J_total)
plt.xlim(0, max(k_J))
plt.ylim(0, max(J_total))
plt.grid(True)
plt.xlabel('Iterasi')
plt.ylabel('Total Jarak Euclidean')
plt.show()
'''
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
