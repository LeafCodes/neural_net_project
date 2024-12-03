import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer

#Deklarasi Data
os.chdir('/Users/ASUS/Downloads/skripsi/data')
data = pd.read_csv('datalemari(-1,1).csv', sep=',', header=0)
data.head()

desired_target          = data.iloc[:,2].values.T                                             
x_input                 = data.iloc[:,:2].values
N                       = x_input.shape[0]                                                          #Jumlah data
x_input                 = np.append(np.ones((N, 1)), x_input, axis=-1).T
i_total                 = x_input.shape[0]                                                          #Jumlah Neuron Input
weight                  = np.array([[0.58748977,    0.45072941,    0.36586638 ]])            
alpha                   = 0.1                                      
k                       = 0
galat_minimum           = 0.0001                                                                    
e_total                 = {'e_w0' : np.array([]), 'e_w1' : np.array([]), 'e_w2' : np.array([])}     #menyimpan error untuk tiap w
k_total                 = {'k_w0' : np.array([]), 'k_w1' : np.array([]), 'k_w2' : np.array([])}     #menyimpan iterasi ke sekian untuk w ke sekian untuk keperluan plotting
t_k                     = {'tk0' : np.array([]), 'tk1' : np.array([]), 'tk2' : np.array([])}        #untuk menandakan iterasi terakhir sebelum berganti w
J_total                 = []
error                   = []                                                                        #menyimpan error keseluruhan
w_total                 = {'w0' : np.array([]), 'w1' : np.array([]), 'w2' : np.array([])}
weight_awal             = np.array([[0.58748977,    0.45072941,    0.36586638 ]])    
print('weight awal      :\n', weight_awal)
print('desired target   :\n', desired_target)                                                                        

#Jarak Euclidean Kuadrat
def J (y_, d_):
    return (np.sum((d_ - y_)** 2))

#Normalisasi data
def minmax_scaling (x_):
    min = np.min(x_)
    max = np.max(x_)
    return ((x_-min)/(max-min))

#Fungsi untuk plot grafik
def func (w, x):
    output = - ((w[0,1]*x/w[0,2])+(w[0,0]/w[0,2]))
    return output

#Forward bias
def forward (w, x) :
    output = np.dot(w, x)
    return output

#Fungsi Turunan Pertama dari jarak euclidean kuadrat
def dJ_dw (x, d, y_pred, i):
    output = 2*np.dot((d-y_pred), x[i].T)
    return output

#Fungsi Turunan Kedua dari jarak euclidean
def D2J_dw (x):
    output = np.sum(-2 * x**2, axis=0)
    return output

#Step Function
def step (y_pred):
    data_1      = y_pred > 0
    data_2      = y_pred < 0
    result_1    = 1 * data_1
    result_2    = -1 * data_2
    result      = result_1 + result_2
    return result

#Fungsi untuk melakukan training bobot
def train (x, w, d, i_, alpha, e, k, J_total, error):
    alpha = alpha/D2J_dw(x)
    for i in range (i_):  
        while True:
            k                   += 1

            #gradient descent
            y_pred              = forward(w, x)
            dj_dwi              = dJ_dw(x, d, y_pred, i)
            w_t                 = w[0, i] - (alpha[i]*dj_dwi)
            e_new               = np.abs((w_t-w[0, i])/w_t)
            w[0, i]             = w_t
            y_pred              = forward(w, x)

            #menyimpan variabel yang dibutuhkan untuk analisa
            e_total[f'e_w{i}']  = np.append(e_total[f'e_w{i}'], e_new)
            k_total[f'k_w{i}']  = np.append(k_total[f'k_w{i}'], k)
            J_total             = np.append(J_total, J(y_pred, d))
            error               = np.append(error, e_new)
            w_total['w0']       = np.append(w_total['w0'], w[0,0])
            w_total['w1']       = np.append(w_total['w1'], w[0,1])
            w_total['w2']       = np.append(w_total['w2'], w[0,2])

            #untuk memberhentikan iterasi
            if e > e_new:
                t_k[f'tk{i}']   = np.append(t_k[f'tk{i}'], k)
                break
    return w, k, y_pred, J_total, error

#Pre-Processing Data
x_input[1:3, :]         = minmax_scaling(x_input[1:3, :])
print('x (norm)         :\n', x_input[1:3, :])
J_awal                  = J(forward(weight, x_input), desired_target)

#Training Data
start = timer()
while True:
    weight, k, ypred, J_total, error    = train (x_input, weight, desired_target, i_total, alpha, galat_minimum, k, J_total, error)
    f                                   = step  (ypred)
    if (np.sum(np.abs(f-desired_target))) == 0 :
        break
end = timer ()
print('Waktu Training   :\n', end-start)    
print('w akhir          :\n', weight)
print('Jumlah iterasi   :\n', k)
print('Prediksi y       :\n', ypred)

#export hasil
hasil = {
    'error' : error,
    'Jarak Euclidean Kuadrat' : J_total,
    'w0' : w_total['w0'],
    'w1' : w_total['w1'],
    'w2' : w_total['w2']
}
df = pd.DataFrame(hasil)
excel_file_path = 'hasil.xlsx'
df.to_excel(excel_file_path, index=False)

#Plotting Grafik
sumbu_x     = np.array([[0], [2.5]])
garis       = func(weight, sumbu_x)

plt.figure(1)
plt.plot(x_input[1, 0:5], x_input[2, 0:5], 'or', label='lemari')
plt.plot(x_input[1, 5:12], x_input[2, 5:12], 'ob', label='buffet')
plt.plot(x_input[1, 12:16], x_input[2, 12:16], 'og', label='wardrobe')
plt.plot(sumbu_x, garis, '-r', label='garis lemari vs semua')
plt.xlabel('Lebar')
plt.ylabel('Tinggi')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Neural Network')
plt.legend()

plt.figure(2)
plt.grid()
plt.plot(k_total['k_w0'], e_total['e_w0'], 'o-', label='w0')
plt.plot(k_total['k_w1'], e_total['e_w1'], 'o-', label='w1')
plt.plot(k_total['k_w2'], e_total['e_w2'], 'o-', label='w2')
plt.xlim(0, k)
plt.ylim(bottom = 0)
plt.xticks(t_k['tk0'], rotation = 45)
max_error = np.append(max(e_total['e_w0']), max(e_total['e_w1']))
max_error = np.append(max_error, max(e_total['e_w2']))
max_error = np.sort(max_error)
full_1 = int(round(max_error[2], 1)*10)
full_2 = np.append([i/100 for i in range (full_1, full_1*10, full_1)], max_error[2])
full_2 = np.append(full_2, max_error[1])
full_2 = np.append(full_2, max_error[0])
plt.yticks(full_2)
plt.xlabel('Iterasi')
plt.ylabel('Error')
plt.title('|W_baru-W_lama|/W_baru')
plt.legend()

plt.figure(3)
plt.grid()
plt.plot(np.append(J_awal, J_total))
plt.xlim(0, k)
plt.ylim(bottom = 0)
plt.xlabel('Iterasi')
plt.ylabel('Error')
plt.title('Jarak Euclidean Kuadrat')
full = J_awal
full_1 = int(round(full, 1)*10)
full_2 = np.append([i/100 for i in range (full_1, full_1*10, full_1)], full)
plt.yticks(full_2)

fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')

ax.plot(weight_awal[0,0], weight_awal[0,1], weight_awal[0,2], c='g', marker='o')
ax.plot(weight[0,0], weight[0,1], weight[0,2], c='r', marker='o')
ax.plot(np.append(weight_awal[0,0], w_total['w0']), np.append(weight_awal[0,1], w_total['w1']), np.append(weight_awal[0,2], w_total['w2']), color='b', linestyle='-', linewidth=2)

ax.set_xlabel('w0')
ax.set_ylabel('w1')
ax.set_zlabel('w2')


"""plt.figure(3)
plt.grid()
plt.plot(ke_total['ke1'], e_total['e1'], 'o-')
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.xticks(t_k['tk1'], rotation = 45)
full = max(e_total['e1'])
full_1 = int(round(full, 1)*10)
full_2 = np.append([i/100 for i in range (full_1, full_1*10, full_1)], full)
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
full = max(e_total['e2'])
full_1 = int(round(full, 1)*10)
full_2 = np.append([i/100 for i in range (full_1, full_1*10, full_1)], full)
plt.yticks(full_2)
plt.xlabel('Iterasi')
plt.ylabel('Error')
plt.title('Error w2')"""
plt.show()