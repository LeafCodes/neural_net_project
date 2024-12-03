import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import sys

#Jarak Euclidean Kuadrat
def J (y_t, d_t):
    return (np.sum((d_t - y_t)** 2, axis=-1))

def J_tot (y_t, d_t):
    return (np.sum((d_t - y_t)** 2))

#Normalisasi data
def minmax_scaling (x_t):
    min = np.min(x_t)
    max = np.max(x_t)
    return ((x_t-min)/(max-min))

#Step Function
def step (y_t):
    data_1      = y_t > 0
    data_2      = y_t < 0
    result_1    = 1 * data_1
    result_2    = -1 * data_2
    result      = result_1 + result_2
    return result

#Sigmoid function
def sigmoid(x_t):
    return 1 / (1 + np.exp(-x_t))

#Relu
def relu(x_t):
    return np.maximum(0, x_t)

#Fungsi untuk plot grafik
def func (w, x, y, j):
    output = - ((w[j,1]*x/w[j,3])+(w[j,2]*y/w[j,3])+(w[j,0]/w[j,3]))
    return output

#turunan pertama fungsi sigmoid
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

#Pembaharuan bobot w1
def dJj_dw1ij (F0_t, d_t, y_t, i_t, j_t):
    output = -2*np.dot((d_t[j_t,:]-y_t[j_t,:]), F0_t[i_t].T)
    return output

def D2J_dw1 (F0_t):
    output = np.sum(2 * F0_t**2, axis=0)
    return output

#Pembaharuan bobot w0
def dj_dw0ij (Y0_t, d_t, y_t, w1_t, x_t, lok_J_max_t, i_t, j_t):
    output = -2 * np.dot(d_t[lok_J_max_t, :] - y_t[lok_J_max_t, :], sigmoid_derivative(Y0_t[j_t].T) * x_t[i_t].T) * w1_t[lok_J_max_t, j_t]
    return output

#Deklarasi Data
os.chdir('/Users/ASUS/Downloads/skripsi/data')
data = pd.read_csv('datalemari(-1,1).csv', sep=',', header=0)
data.head()

desired_target          = data.iloc[:,2:5].values.T                                             
x_input                 = data.iloc[:,:2].values
x_input                 = np.append(np.ones((len(x_input), 1)), x_input, axis=-1).T
x_input[1:3, :]         = minmax_scaling(x_input[1:3, :])
jumlah_input_neuron     = x_input.shape[0]
jumlah_hidden_neuron    = 2
jumlah_output_neuron    = len(desired_target)
weight_0                = np.random.rand(jumlah_hidden_neuron, jumlah_input_neuron)
weight_1                = np.random.rand(jumlah_output_neuron, jumlah_hidden_neuron+1)
epoch                   = 0
F0_total                = []
alpha_const             = 0.1
k                       = 0
galat                   = 0.001

#forward bias
Y0 = np.dot(weight_0, x_input)
F0 = sigmoid(Y0)
F0_total = F0
F0 = np.append(np.ones((1,16)), F0, axis=0)
Y1 = np.dot(weight_1, F0)
F1 = step(Y1)
J_total = J(Y1, desired_target).reshape(1, 3)
J_total_max = J_tot(Y1, desired_target)

start = timer()
while (True):
    #Pembaharuan bobot w1
    for j in range(jumlah_output_neuron):
        for i in range (jumlah_hidden_neuron+1):
            while (True):
                k       += 1
                alpha   = alpha_const
                dj_dw1  = []
                d2J_dw1 = []

                Y1      = np.dot(weight_1, F0)
                J_      = J(Y1, desired_target).reshape(1, 3)
                J_total = np.append(J_total, J_, axis=0)
                J_total_max = np.append(J_total_max, J_tot(Y1, desired_target))
                dj_dw1  = dJj_dw1ij(F0, desired_target, Y1, i, j)
                d2J_dw1 = D2J_dw1(F0[i])
                alpha   = alpha/d2J_dw1
                w1_new  = weight_1[j,i] - alpha * dj_dw1
                if np.abs((w1_new-weight_1[j,i])/w1_new) < galat :
                    weight_1[j,i] = w1_new
                    break
                weight_1[j,i] = w1_new

    #Pembaharuan W0
    for j in range(jumlah_hidden_neuron):
        for i in range (jumlah_input_neuron):
            while (True):
                k       += 1
                alpha   = alpha_const
                dj_dw0  = []

                Y0      = np.dot(weight_0, x_input)
                F0      = sigmoid(Y0)
                F0      = np.append(np.ones((1,16)), F0, axis=0)
                Y1      = np.dot(weight_1, F0)

                J_      = J(Y1, desired_target).reshape(1, 3)
                J_total = np.append(J_total, J_, axis=0)
                J_total_max = np.append(J_total_max, J_tot(Y1, desired_target))
                dj_dw0  = dj_dw0ij(Y0, desired_target, Y1, weight_1, x_input, 0, i, j) + dj_dw0ij(Y0, desired_target, Y1, weight_1, x_input, 1, i, j) + dj_dw0ij(Y0, desired_target, Y1, weight_1, x_input, 2, i, j)
                w0_new  = weight_0[j,i] - alpha * dj_dw0
                weight_0[j,i] = w0_new
                if (np.abs((w0_new-weight_0[j,i])/w0_new)) < galat :
                    weight_0[j,i] = w0_new
                    break
                

    epoch += 1
    F0_total= np.append(F0_total, F0[1:3, :], axis=0)
    if np.sum(np.abs(step(Y1) - desired_target)) == 0 :
        Total_time = timer() - start
        break


    if timer()-start > 10 :
        Total_time = timer() - start
        break

#menyimpan data
#os.chdir('/Users/ASUS/Downloads/skripsi/data/1 hl 3 neuron/run 10')
np.savetxt('nilai f0.csv', F0_total, delimiter=',')
np.savetxt('nilai w0.csv', weight_0, delimiter=',')
np.savetxt('nilai w1.csv', weight_1, delimiter=',')
np.savetxt('nilai J_total.csv', J_total_max, delimiter=',')
np.savetxt('nilai J.csv', J_total, delimiter=',')

with open('data.txt', 'w') as output_file :
    sys.stdout = output_file
    print('Total time :', Total_time)
    print('Iterasi/sec :', k/Total_time)
    print('Jumlah iterasi :', k)
    print('Jumlah Epoch :', epoch)

sys.stdout = sys.__stdout__

'''Plotting Grafik
x             = np.linspace(min(F0[1,:]), max(F0[1 ,:]), 10)
y             = np.linspace(min(F0[2, :]), max(F0[2, :]), 10)
X, Y          = np.meshgrid(x, y)
garis_0       = func(weight_1, X, Y, 0)
garis_1       = func(weight_1, X, Y, 1)
garis_2       = func(weight_1, X, Y, 2)

sumbu_x_max   = round(np.max(F0[1, :]), 3)+0.01
sumbu_y_max   = round(np.max(F0[2, :]), 3)+0.01

plt.figure(1)
plt.plot(x_input[1, 0:5], x_input[2, 0:5], 'or', label='lemari')
plt.plot(x_input[1, 5:12], x_input[2, 5:12], 'ob', label='buffet')
plt.plot(x_input[1, 12:16], x_input[2, 12:16], 'og', label='wardrobe')
plt.xlabel('Lebar')
plt.ylabel('Tinggi')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Plot input X')
plt.legend()
plt.savefig('Plot input X.png')

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')

ax.plot(F0[1, 0:5],   F0[2, 0:5],   F0[3, 0:5],   'or', label='lemari')
ax.plot(F0[1, 5:12],  F0[2, 5:12],  F0[3, 5:12],  'ob', label='buffet')
ax.plot(F0[1, 12:16], F0[2, 12:16], F0[3, 12:16], 'og', label='wardrobe')
ax.plot_surface(X, Y, garis_0)
ax.plot_surface(X, Y, garis_1)
ax.plot_surface(X, Y, garis_2)
ax.legend()
plt.savefig('Plot F1 dan W.png')

plt.figure(3)
plt.plot(J_total[:, 0], label='J0')
plt.plot(J_total[:, 1], label='J1')
plt.plot(J_total[:, 2], label='J2')
plt.xlim(-1, k)
plt.ylim(0, np.max(J_total))
plt.title('J')
plt.xlabel('iterasi ke')
max_J = np.array([max(J_total[:, 0]), max(J_total[:, 1]), max(J_total[:, 2])])
max_J = np.sort(max_J)
full_1 = int(round(max_J[2], 1)*10)
full_2 = np.append([i/100 for i in range (full_1, full_1*10, full_1)], max_J[2])
full_2 = np.append(full_2, max_J[1])
full_2 = np.append(full_2, max_J[0])
plt.yticks(full_2)
plt.legend()
plt.savefig('Plot J')

print('Total time :', Total_time)
plt.show()'''