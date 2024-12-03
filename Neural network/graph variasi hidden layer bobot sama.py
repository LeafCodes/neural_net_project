import numpy as np
from matplotlib import pyplot as plt
from polynom_utils import polynom_curve_fit
import os

list_bobot = {}
hidden_layers = {}
x_axis = {}
hidden_layer_axis = {}
for i in range (1, 5):
    hidden_layer_axis[i] = []
    x_axis[i] = []

jumlah_bobot = [72, 96, 105, 108, 126, 138, 144, 156, 162, 180, 186, 189, 198, 216, 222, 234, 240, 246, 252, 264, 270, 276, 288, 360]
list_bobot[72] = ['4-4-4-4', '4-6-4', '6-6', '12']
list_bobot[96] = ['4-4-5-6', '5-5-7', '4-12', '16']
list_bobot[105] = ['5-5-5-5', '7-4-8']
list_bobot[108] = ['4-5-4-8', '4-6-8', '6-10', '18']
list_bobot[126] = ['4-4-7-7', '5-6-9', '6-12', '21']
list_bobot[138] = ['4-5-8-6', '11-3-12', '4-18', '23']
list_bobot[144] = ['6-6-6-6', '3-21-3', '6-14', '24']
list_bobot[156] = ['4-4-7-10', '6-9-7', '8-12', '26']
list_bobot[162] = ['4-8-5-10', '9-6-9', '6-16', '27']
list_bobot[180] = ['4-8-7-8', '8-9-7', '6-18', '30']
list_bobot[186] = ['4-5-10-8', '15-3-16', '10-12','31']
list_bobot[189] = ['7-7-7-7', '10-6-11', '6-19']
list_bobot[198] = ['4-4-10-10', '9-8-9', '6-20', '33']
list_bobot[216] = ['4-9-10-6', '9-9-9', '6-22', '36']
list_bobot[222] = ['4-10-7-10', '18-3-19', '8-18', '37']
list_bobot[234] = ['6-7-10-8', '9-10-9', '6-24', '39']
list_bobot[240] = ['8-8-8-8', '10-9-10', '40']
list_bobot[246] = ['9-9-8-6', '20-3-21', '12-14', '41']
list_bobot[252] = ['5-8-10-9', '11-9-10', '6-26', '42']
list_bobot[264] = ['6-10-9-8', '11-9-11', '10-18', '44']
list_bobot[270] = ['7-7-10-10', '9-12-9', '6-28', '45']
list_bobot[276] = ['9-9-10-6', '11-9-12', '12-16', '46']
list_bobot[288] = ['6-10-9-10', '9-13-9', '8-24', '48']
list_bobot[360] = ['10-10-10-10', '11-12-13', '6-38', '60']

for i in jumlah_bobot:
    hidden_layers[i] = []
    for j in list_bobot[i]:
        hidden_layers[i].append(list(map(int, j.split('-'))))
        x_axis[len(hidden_layers[i][-1])].append(i)

plt.figure(figsize=(12,7))
for i, n in enumerate (jumlah_bobot):
    for j in list_bobot[n]:    
        y_axis = np.load(f'D:/skripsi/data hasil/Variasi jumlah layer dengan bobot yang sama/{n}/3-{j}-3/mse_epoch.npy')
        y_axis = len(y_axis)
        plt.scatter(jumlah_bobot[i], y_axis)

plt.xticks(jumlah_bobot)
#plt.show()

plt.figure(figsize=(20,7))
for i, n in enumerate (jumlah_bobot):
    for k, j in enumerate (list_bobot[n]):
        length = len(hidden_layers[n][k])    
        y_axis = np.load(f'D:/skripsi/data hasil/Variasi jumlah layer dengan bobot yang sama/{n}/3-{j}-3/mse_epoch.npy')
        y_axis = len(y_axis)

        hidden_layer_axis[length].append(y_axis)

for i in range (1, 5) :
    plt.scatter(x_axis[i], hidden_layer_axis[i])
    plt.plot(x_axis[i], hidden_layer_axis[i], label=f'{i} Hidden Layer')

# Menambahkan grid dan mengatur xticks
plt.xticks(rotation=45, fontsize=10)  # Mengubah rotasi dan ukuran font agar terbaca lebih jelas
plt.xlim(70, 370)
plt.xlabel('Jumlah Bobot')
plt.ylabel('Jumlah Iterasi')
plt.ylim(0, 2500)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Membuat jarak xticks sama
plt.legend()
# Menampilkan plot yang sudah diperbarui
plt.tight_layout()  # Agar plot lebih rapi dan tidak terpotong
plt.grid()
plt.show()