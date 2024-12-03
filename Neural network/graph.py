import numpy as np
from matplotlib import pyplot as plt
from polynom_utils import polynom_curve_fit
import os

learning_rate = {}
for i in np.arange (0.1, 1.1, 0.1):
  learning_rate[f'{i:.1f}'] = []
  for j in range(3, 11):
    temp = np.load(f'D:/skripsi/data hasil/Variasi 1 Hidden Layer/Learning rate = {i:.1f}, weight init = 0.3/3-{j}-3/mse_epoch.npy')
    learning_rate[f'{i:.1f}'].append(len(temp))

mean = []
std = []
for i in learning_rate:
  mean.append(np.mean(learning_rate[i]))
  std.append(np.std(learning_rate[i]))

plt.figure(figsize = (10, 5))
plt.scatter(np.arange(0.1, 1.1, 0.1), mean)
plt.plot(np.arange(0.1, 1.1, 0.1), mean, label='Mean')
plt.scatter(np.arange(0.1, 1.1, 0.1), std)
plt.plot(np.arange(0.1, 1.1, 0.1), std, label='Standard Deviasi')
plt.legend()
plt.xticks(np.arange(0.1, 1.1, 0.1))
plt.xlabel('Learning Rate')
plt.ylabel('Mean')
plt.xlim(0, 1.1)
plt.ylim(bottom = 0)
plt.grid()
plt.show()

weight_init = {}
for i in np.arange (0.1, 1.1, 0.1):
  weight_init[f'{i:.1f}'] = []
  for j in range(3, 11):
    temp = np.load(f'D:/skripsi/data hasil/Variasi 1 Hidden Layer/Learning rate = 0.1, weight init = {i:.1f}/3-{j}-3/mse_epoch.npy')
    weight_init[f'{i:.1f}'].append(len(temp))

mean = []
std = []
for i in weight_init:
  mean.append(np.mean(weight_init[i]))
  std.append(np.std(weight_init[i]))

plt.figure(figsize = (10, 5))
plt.scatter(np.arange(0.1, 1.1, 0.1), mean)
plt.plot(np.arange(0.1, 1.1, 0.1), mean, label='Mean')
plt.scatter(np.arange(0.1, 1.1, 0.1), std)
plt.plot(np.arange(0.1, 1.1, 0.1), std, label='Standard Deviasi')
plt.legend()
plt.xticks(np.arange(0.1, 1.1, 0.1))
plt.xlabel('Bobot Initial')
plt.ylabel('Mean')
plt.xlim(0, 1.1)
plt.ylim(bottom = 0)
plt.grid()
plt.show()

y_axis = []
for i in range(1,11):
  path = (f'D:/skripsi/data hasil/Variasi 1 Hidden Layer/Learning rate = 0.1, weight init = 0.3/3-{i}-3/mse_epoch.npy')
  y_axis.append(len(np.load(path)))

x_axis= list(range(1,11))

plt.figure(figsize=(12,7))
plt.plot(x_axis,y_axis)
plt.scatter(x_axis[0:2], y_axis[0:2], label='Tidak Mencapai 100% Akurasi')
plt.scatter(x_axis[2:11], y_axis[2:11], label='Mencapai 100% Akurasi')
plt.title('Pengaruh Jumlah Hidden Neuron terhadap Jumlah Iterasi pada Arsitektur Satu Hidden Layer [3-n-3]', size=16)
plt.xlabel('Jumlah Hidden Neuron (n)', size=14)
plt.ylabel('Jumlah Iterasi',size=14)
plt.xticks(x_axis, size = 12)
plt.yticks(size =12)
plt.ylim(bottom=0)
plt.grid()
plt.legend()
plt.show()

orde = 2

x_axis = list(range(1,11))
x_axis_t, y_axis_t = polynom_curve_fit(x_axis[2:11], y_axis[2:11], orde)

plt.figure(figsize=(12,7))
plt.plot(x_axis[2:11], y_axis[2:11])
plt.scatter(x_axis[2:11], y_axis[2:11])
plt.plot(x_axis_t, y_axis_t, label=f'Polynomial Curve fit Orde {orde}')
plt.title('Pengaruh Jumlah Hidden Neuron terhadap Jumlah Iterasi pada Arsitektur Satu Hidden Layer [3-n-3]', size=16)
plt.xlabel('Jumlah Hidden Neuron (n)', size=14)
plt.ylabel('Jumlah Iterasi',size=14)
plt.xticks(x_axis[2:11], size = 12)
plt.ylim(0, 250)
plt.yticks(size =12)
plt.legend()
plt.grid()
plt.show()

y_axis = []
x_axis= list(range(1,11))
for i in range (1,11):
  path = (f'D:/skripsi/data hasil/Variasi 1 Hidden Layer/Learning rate = 0.1, weight init = 0.3/3-{i}-3/accuracy_epoch.npy')
  y_axis.append(np.load(path)[-1])

plt.figure(figsize=(12,7))
plt.scatter(x_axis[0:2], y_axis[0:2], label='Tidak Mencapai 100% Akurasi')
plt.scatter(x_axis[2:11], y_axis[2:11], label='Mencapai 100% Akurasi')
plt.plot(x_axis, y_axis)
plt.title('Pengaruh Jumlah Perceptron terhadap Akurasi pada Arsitektur Satu Hidden Layer [3-n-3]', size=16)
plt.xlabel('Jumlah Perceptron (n)', size=14)
plt.ylabel('Akurasi (%)',size=14)
plt.xticks(x_axis, size = 12)
plt.yticks(size =12)
plt.grid()
plt.legend()
plt.show()

y_axis = []
x_axis= list(range(1,11))

for n in range (1,11):
  path = (f'D:/skripsi/data hasil/Variasi 1 Hidden Layer/Learning rate = 0.1, weight init = 0.3/3-{n}-3/mse.npy')
  y_axis.append(np.load(path)[-1])

plt.figure(figsize=(12,7))
plt.scatter(x_axis, y_axis)
plt.plot(x_axis, y_axis)

plt.title('Pengaruh Jumlah Hidden Neuron terhadap MSE Akhir pada Arsitektur Satu Hidden Layer [3-n-3]', size=16)
plt.xlabel('Jumlah Hidden Neuron (n)', size=14)
plt.ylabel('MSE Akhir',size=14)
plt.xticks(x_axis, size = 12)
plt.yticks(size =12)
plt.grid()
plt.show()

orde = 2

x_axis_t, y_axis_t = polynom_curve_fit(x_axis[2:11], y_axis[2:11], orde)

plt.figure(figsize=(12,7))
plt.scatter(x_axis[2:11], y_axis[2:11])
plt.plot(x_axis[2:11], y_axis[2:11])
plt.plot(x_axis_t, y_axis_t, label=f'Polynomial Curve fit Orde {orde}')
plt.title('Pengaruh Jumlah Hidden Neuron terhadap MSE Akhir pada Arsitektur Satu Hidden Layer [3-n-3]', size=16)
plt.xlabel('Jumlah Hidden Neuron (n)', size=14)
plt.ylabel('MSE Akhir',size=14)
plt.xticks(x_axis[2:11], size = 12)
plt.yticks(size =12)
plt.ylim(0, 0.3)
plt.grid()
plt.legend()
plt.show()

x_axis= list(range(1,11))

y_axis = []
y_axis1 = []
for n in range (1,11):
  path = (f'D:/skripsi/data hasil/Variasi 1 Hidden Layer/Learning rate = 0.1, weight init = 0.3/3-{n}-3/mse_epoch.npy')
  y_axis.append(np.load(path)[-1])
  y_axis1.append(len(np.load(path)))

# Membuat figure dan axis
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(x_axis[2:11], y_axis[2:11], color='blue', label='MSE Akhir')
ax1.scatter(x_axis[2:11], y_axis[2:11] , color='blue')
ax1.set_xlabel('Jumlah Hidden Neuron (n)')
ax1.set_ylabel('MSE Akhir')
ax1.tick_params(axis='y', labelcolor='blue')


ax2 = ax1.twinx()
ax2.plot(x_axis[2:11], y_axis1[2:11], color='maroon', label='Jumlah Iterasi')
ax2.scatter(x_axis[2:11], y_axis1[2:11], color='maroon')
ax2.set_ylabel('Jumlah Iterasi', color='maroon')
ax2.tick_params(axis='y', labelcolor='maroon')

# Menambahkan legenda
fig.legend(loc='upper left')

# Menampilkan plot
plt.show()

for n in range (1, 11):
  fig, ax1 = plt.subplots()
  path = (f'D:/skripsi/data hasil/Variasi 1 Hidden Layer/Learning rate = 0.1, weight init = 0.3/3-{n}-3')
  temp = os.path.join(path, 'mse_epoch.npy')
  ax1.plot(np.load(temp), color='red', label='MSE')
  ax1.set_ylabel('MSE', color='red')
  ax2 = ax1.twinx()
  temp = os.path.join(path, 'accuracy_epoch.npy')
  print(path)
  ax2.plot(np.load(temp), label='Akurasi')
  ax1.set_title(f'Plot (3-{n}-3)')
  ax1.set_xlabel('Iterasi Ke')
  ax2.set_ylabel('Akurasi (%)', color='blue')
  ax1.legend(loc='upper left')
  ax2.legend(loc='upper right')
  ax1.set_ylim(0,3)
  ax2.set_ylim(0,100)
  plt.xlim(left=-0.5)
  plt.grid()
  plt.show()