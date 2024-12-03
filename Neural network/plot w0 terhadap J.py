import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

#Jarak Euclidean Kuadrat
def J (y_t, d_t):
    return np.sum((d_t - y_t)** 2)

#Normalisasi data
def minmax_scaling (x_t):
    min = np.min(x_t)
    max = np.max(x_t)
    return ((x_t-min)/(max-min))

#Sigmoid function
def sigmoid(x_t):
    return 1 / (1 + np.exp(-x_t))

#Step Function
def step (y_t):
    data_1      = y_t > 0
    data_2      = y_t < 0
    result_1    = 1 * data_1
    result_2    = -1 * data_2
    result      = result_1 + result_2
    return result

os.chdir('/Users/ASUS/Downloads/skripsi/data')
# Membaca file CSV
w0 = pd.read_csv('nilai w0.csv', header=None).values
w1 = pd.read_csv('nilai w1.csv', header=None).values
data = pd.read_csv('datalemari(-1,1).csv', sep=',', header=0)
data.head()
desired_target          = data.iloc[:,2:5].values.T                                             
x_input                 = data.iloc[:,:2].values
x_input                 = np.append(np.ones((len(x_input), 1)), x_input, axis=-1).T
x_input[1:3, :]         = minmax_scaling(x_input[1:3, :])

#forward bias
Y0          = np.dot(w0, x_input)
F0          = sigmoid(Y0)
F0          = np.append(np.ones((1,16)), F0, axis=0)
Y1          = np.dot(w1, F0)
print("J :\n", np.sum((desired_target - Y1)** 2, axis=-1))
def plot_w0 (weight_0, i, j, d, weight_1, x):
    jumlah_sebelum = 50
    jumlah_sesudah = 50
    step = 0.1

    sebelum = [weight_0[j,i] - k * step for k in range(jumlah_sebelum, 0, -1)]
    sesudah = [weight_0[j,i] + k * step for k in range(1, jumlah_sesudah + 1)]
    
    sumbux = sebelum + [weight_0[j,i]] + sesudah

    J_ = np.zeros(len(sumbux))

    for k in range (len(sumbux)):
        w0_             = pd.read_csv('nilai w0.csv', header=None).values
        w0_[j, i]       = sumbux[k]
        Y0_             = np.dot(w0_, x)
        F0_             = sigmoid(Y0_)
        F0_             = np.append(np.ones((1,16)), F0_, axis=0)
        Y1_             = np.dot(weight_1, F0_)
        J_[k]           = J(Y1_, d)
    return J_, sumbux

k = 1
print('W0 :\n', w0)
for i_ in range (3):
    for j_ in range(2):
        plt.figure(k)
        plt.plot(plot_w0(w0, i_, j_, desired_target, w1, x_input)[1], plot_w0(w0, i_, j_, desired_target, w1, x_input)[0])
        plt.plot(plot_w0(w0, i_, j_, desired_target, w1, x_input)[1][50], plot_w0(w0, i_, j_, desired_target, w1, x_input)[0][50], 'or')
        plt.title(f"W0 ({j_},{i_}) terhadap J{j_}")
        k += 1

plt.show()