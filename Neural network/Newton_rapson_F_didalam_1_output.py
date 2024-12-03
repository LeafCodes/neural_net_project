import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer

#Deklarasi Data
os.chdir('/Users/ASUS/Downloads/skripsi/data')
data = pd.read_csv('datalemari(-1,1).csv', sep=',', header=0)
data.head()

d           = data.iloc[:,2].values.T                                             
x           = data.iloc[:,:2].values
N           = x.shape[0]
x           = np.append(np.ones((N, 1)), x, axis=-1).T
i_total     = x.shape[0]
w           = np.array([[0.5874897, 0.45072941, 0.36586638]])                
alpha       = 0.1                                      
k           = 0
e           = 0.0001
e_total     = {'e0' : np.array([]), 'e1' : np.array([]), 'e2' : np.array([])}
ke_total    = {'ke0' : np.array([]), 'ke1' : np.array([]), 'ke2' : np.array([])}
t_k         = {'tk0' : np.array([]), 'tk1' : np.array([]), 'tk2' : np.array([])}

print('x :', x)
print('w :', w)


#Deklarasi Fungsi
def func (w, x):
    y = - ((w[0,1]*x/w[0,2])+(w[0,0]/w[0,2]))
    return y

#Fungsi J
def J (ypred, d):
    result = np.sum((d-ypred) ** 2)
    return result
#Fungsi F
def sigmoid (x):
    result = 1/(1+np.exp(-x))
    return result

def derivative_sigmoid(x):
    result = sigmoid(x) * (1-sigmoid(x))
    return result

def sec_deriv_sig(x):
    result = (np.exp(-x)-np.exp(-2*x))/((1+np.exp(-x)) ** 3)
    return result

def step (x) :
    data_1      = x > 0.5
    data_2      = x < 0.5
    result_1    = 1 * data_1
    result_2    = 0 * data_2
    result      = result_1 + result_2
    return result


def gantijadi0 (x):
    data_1      = x > 0
    data_2      = x < 0
    result_1    = 1 * data_1
    result_2    = 0 * data_2
    result      = result_1 + result_2
    return result

d           = gantijadi0(d)
print('d :', d)

#Forward bias
def forward (w,x) :
    y = np.dot(w,x)
    return y

#Fungsi Turunan Pertama dari jarak euclidean
def dJ_dw (x, d, y_pred, f,  i_):
    output = -2 * np.dot(((d-f) * derivative_sigmoid(y_pred)), x[i_].T)
    return output

#Fungsi Turunan Kedua dari jarak euclidean
def D2J_dw (x, y_pred, f, i_):
    output = -2 * np.dot((-(derivative_sigmoid(y_pred) ** 2)+((d-f) * sec_deriv_sig(y_pred))), x[i_].T ** 2)
    return output

#Fungsi untuk melakukan training bobot
def train (x, w, d, i_, alpha, e, k):
    for i in range (i_):  
        while True:
            k                   += 1
            y_pred              = forward(w, x)
            f                   = sigmoid(y_pred)
            dj_dwi              = dJ_dw(x, d, y_pred, f, i)
            d2j_dwi             = D2J_dw (x, y_pred, f, i)
            w_t                 = w[0, i] - ((alpha/d2j_dwi)*dj_dwi)
            e_new               = np.abs((w_t-w[0, i])/w_t)
            e_total[f'e{i}']    = np.append(e_total[f'e{i}'], e_new)
            ke_total[f'ke{i}']  = np.append(ke_total[f'ke{i}'], k)
            if e > e_new:
                w[0, i] = w_t
                t_k[f'tk{i}']   = np.append(t_k[f'tk{i}'], k)
                break
            w[0, i]             = w_t
    return w, k, y_pred, f

#main function
f                   = sigmoid(forward(w, x))
error               = J(f, d)

start                           = timer()
while True:
    w, k, ypred, f              = train (x, w, d, i_total, alpha, e, k)
    error                       = J(f, d)
    threshold                   = step(f)
    if np.sum(threshold-d) == 0 :
        break
end                             = timer()

print('f :', f)
print('Waktu Komputasi :', end-start)
print('w akhir = ', w)
print('Jumlah iterasi :', k)

sumbu_x = np.array([[0], [2.5]])
garis = func(w, sumbu_x)

plt.figure(1)
plt.plot(x[1, 0:5], x[2, 0:5], 'or', label='lemari')
plt.plot(x[1, 5:12], x[2, 5:12], 'ob', label='buffet')
plt.plot(x[1, 12:16], x[2, 12:16], 'og', label='wardrobe')
plt.plot(sumbu_x, garis, '-r', label='garis 1')
plt.xlabel('Lebar')
plt.ylabel('Tinggi')
plt.xlim(0.5, 2.5)
plt.ylim(0, 2.5)
plt.title('Neural Network')
plt.legend()

plt.figure(2)
plt.grid()
plt.plot(ke_total['ke0'], e_total['e0'], 'o-')
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.xticks(t_k['tk0'], rotation = 45)
full    = max(e_total['e0'])
full_1  = int(round(full, 1)*10)
full_2  = np.append([i/100 for i in range (full_1, full_1*10, full_1)], full)
plt.yticks(full_2)
plt.xlabel('Iterasi')
plt.ylabel('Error')
plt.title('Error w0')

plt.figure(3)
plt.grid()
plt.plot(ke_total['ke1'], e_total['e1'], 'o-')
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.xticks(t_k['tk1'], rotation = 45)
full    = max(e_total['e1'])
full_1  = int(round(full, 1)*10)
full_2  = np.append([i/100 for i in range (full_1, full_1*10, full_1)], full)
plt.yticks(full_2)
plt.xlabel('Iterasi')
plt.ylabel('Error')
plt.title('Error w1')

plt.figure(4)
plt.grid()
plt.plot(ke_total['ke2'], e_total['e2'], 'o-')
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.xticks(t_k['tk2'], rotation = 45)
full    = max(e_total['e2'])
full_1  = int(round(full, 1)*10)
full_2  = np.append([i/100 for i in range (full_1, full_1*10, full_1)], full)
plt.yticks(full_2)
plt.xlabel('Iterasi')
plt.ylabel('Error')
plt.title('Error w2')

plt.show()