import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

#Jarak Euclidean Kuadrat
def J (y_t, d_t):
    return (np.sum((d_t - y_t)** 2, axis=-1))

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
print(J(Y1, desired_target))

def plot_w1 (weight, i, j, F0, d):
    jumlah_sebelum = 10
    jumlah_sesudah = 10
    step = 1

    sebelum = [weight[j,i] - k * step for k in range(jumlah_sebelum, 0, -1)]
    sesudah = [weight[j,i] + k * step for k in range(1, jumlah_sesudah + 1)]

    sumbux = sebelum + [weight[j,i]] +sesudah

    J_ = []
    for k in range (len(sumbux)):
        if j == 0 :
            if i == 0 :
                J_ = np.append(J_, np.sum((d[j, :] - (sumbux[k]*F0[0,:]+weight[0,1]*F0[1,:]+weight[0,2]*F0[2,:]))**2))
            if i == 1 :
                J_ = np.append(J_, np.sum((d[j, :] - (sumbux[k]*F0[1,:]+weight[0,0]*F0[0,:]+weight[0,2]*F0[2,:]))**2))
            if i == 2 :
                J_ = np.append(J_, np.sum((d[j, :] - (sumbux[k]*F0[2,:]+weight[0,0]*F0[0,:]+weight[0,1]*F0[1,:]))**2))
        if j == 1 :
            if i == 0 :
                J_ = np.append(J_, np.sum((d[j, :] - (sumbux[k]*F0[0,:]+weight[j,1]*F0[1,:]+weight[j,2]*F0[2,:]))**2))
            if i == 1 :
                J_ = np.append(J_, np.sum((d[j, :] - (sumbux[k]*F0[1,:]+weight[j,0]*F0[0,:]+weight[j,2]*F0[2,:]))**2))
            if i == 2 :
                J_ = np.append(J_, np.sum((d[j, :] - (sumbux[k]*F0[2,:]+weight[j,0]*F0[0,:]+weight[j,1]*F0[1,:]))**2))
        if j == 2 :
            if i == 0 :
                J_ = np.append(J_, np.sum((d[j, :] - (sumbux[k]*F0[0,:]+weight[j,1]*F0[1,:]+weight[j,2]*F0[2,:]))**2))
            if i == 1 :
                J_ = np.append(J_, np.sum((d[j, :] - (sumbux[k]*F0[1,:]+weight[j,0]*F0[0,:]+weight[j,2]*F0[2,:]))**2))
            if i == 2 :
                J_ = np.append(J_, np.sum((d[j, :] - (sumbux[k]*F0[2,:]+weight[j,0]*F0[0,:]+weight[j,1]*F0[1,:]))**2))
    return J_, sumbux

plt.figure(1)
plt.plot(plot_w1(w1, 0, 0, F0, desired_target)[1], plot_w1(w1, 0, 0, F0, desired_target)[0])
plt.plot(plot_w1(w1, 0, 0, F0, desired_target)[1][10], plot_w1(w1, 0, 0, F0, desired_target)[0][10], 'or')

plt.figure(2)
plt.plot(plot_w1(w1, 1, 0, F0, desired_target)[1], plot_w1(w1, 1, 0, F0, desired_target)[0])
plt.plot(plot_w1(w1, 1, 0, F0, desired_target)[1][10], plot_w1(w1, 1, 0, F0, desired_target)[0][10], 'or')

plt.figure(3)
plt.plot(plot_w1(w1, 2, 0, F0, desired_target)[1], plot_w1(w1, 2, 0, F0, desired_target)[0])
plt.plot(plot_w1(w1, 2, 0, F0, desired_target)[1][10], plot_w1(w1, 2, 0, F0, desired_target)[0][10], 'or')

plt.figure(4)
plt.plot(plot_w1(w1, 0, 1, F0, desired_target)[1], plot_w1(w1, 0, 1, F0, desired_target)[0])
plt.plot(plot_w1(w1, 0, 1, F0, desired_target)[1][10], plot_w1(w1, 0, 1, F0, desired_target)[0][10], 'or')

plt.figure(5)
plt.plot(plot_w1(w1, 1, 1, F0, desired_target)[1], plot_w1(w1, 1, 1, F0, desired_target)[0])
plt.plot(plot_w1(w1, 1, 1, F0, desired_target)[1][10], plot_w1(w1, 1, 1, F0, desired_target)[0][10], 'or')

plt.figure(6)
plt.plot(plot_w1(w1, 2, 1, F0, desired_target)[1], plot_w1(w1, 2, 1, F0, desired_target)[0])
plt.plot(plot_w1(w1, 2, 1, F0, desired_target)[1][10], plot_w1(w1, 2, 1, F0, desired_target)[0][10], 'or')

plt.figure(7)
plt.plot(plot_w1(w1, 0, 2, F0, desired_target)[1], plot_w1(w1, 0, 2, F0, desired_target)[0])
plt.plot(plot_w1(w1, 0, 2, F0, desired_target)[1][10], plot_w1(w1, 0, 2, F0, desired_target)[0][10], 'or')

plt.figure(8)
plt.plot(plot_w1(w1, 1, 2, F0, desired_target)[1], plot_w1(w1, 1, 2, F0, desired_target)[0])
plt.plot(plot_w1(w1, 1, 2, F0, desired_target)[1][10], plot_w1(w1, 1, 2, F0, desired_target)[0][10], 'or')

plt.figure(9)
plt.plot(plot_w1(w1, 2, 2, F0, desired_target)[1], plot_w1(w1, 2, 2, F0, desired_target)[0])
plt.plot(plot_w1(w1, 2, 2, F0, desired_target)[1][10], plot_w1(w1, 2, 2, F0, desired_target)[0][10], 'or')

plt.show()