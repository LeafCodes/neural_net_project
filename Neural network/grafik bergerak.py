import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from neural_network_class import NeuralNetwork

# Fungsi untuk melakukan scaling min-max
def minmax_scaling(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# Load dan scaling data
data = pd.read_csv('D:/skripsi/data/datalemari(-1,1).csv', sep=',', header=0)
X = minmax_scaling(data.iloc[:, :2].values.T)

# Definisikan rentang nilai untuk setiap sumbu
x_axis = np.linspace(0, 1, 101)
y_axis = np.linspace(0, 1, 101)
X_axis, Y_axis = np.meshgrid(x_axis, y_axis)
XY = np.vstack((X_axis.flatten(), Y_axis.flatten()))
X_test = np.vstack([np.ones(XY.shape[1]), XY])  # Tambahkan bias

# Fungsi untuk menghasilkan dan menyimpan animasi
def generate_animation(learning_rate, weight_init, config):
    NN = NeuralNetwork(2, [config], 3, weight_init=weight_init)

    # Load bobot pelatihan sesuai konfigurasi
    weights = [np.load(f'D:/skripsi/data hasil/Variasi 1 Hidden Layer/Learning rate = {learning_rate}, weight init = {weight_init}/3-{config}-3/weight_total_epoch_{i}.npy') for i in range(2)]

    # Simpan hasil keluaran di setiap frame
    F = [NN.step_function(np.dot(weights[1][i], np.dot(weights[0][i], X_test))) for i in range(len(weights[0]))]
    F = np.array(F)

    # Membuat figure dan axis
    fig, ax = plt.subplots(figsize=(8, 6))
    contours = [ax.contourf(X_axis, Y_axis, F[0][i, :].reshape(X_axis.shape), levels=20, cmap=cmap, alpha=0.1) 
                for i, cmap in enumerate(['viridis', 'plasma', 'inferno'])]
    ax.set(title=f'Output Tiga Neuron\nLR={learning_rate}, WI={weight_init}, Config={config}', xlabel='Lebar (m)', ylabel='Tinggi (m)')
    ax.scatter(X[0][:5], X[1][:5], label='Lemari')
    ax.scatter(X[0][5:12], X[1][5:12], label='Buffet')
    ax.scatter(X[0][12:], X[1][12:], label='Wardrobe')
    ax.legend()

    # Fungsi update untuk setiap frame animasi
    def update(frame):
        for i, cmap in enumerate(['viridis', 'plasma', 'inferno']):
            for c in contours[i].collections: c.remove()
            contours[i] = ax.contourf(X_axis, Y_axis, F[frame][i, :].reshape(X_axis.shape), levels=20, cmap=cmap, alpha=0.1)

    # Animasi
    ani = FuncAnimation(fig, update, frames=len(weights[0]), blit=False, repeat=False)
    
    # Simpan animasi sebagai GIF dengan nama yang sesuai
    save_path = f'animasi_neuron_LR_{learning_rate}_WI_{weight_init}_Config_3-{config}-3.gif'
    ani.save(save_path, writer='pillow')
    plt.close(fig)  # Tutup figure setelah selesai agar tidak menumpuk

# Loop untuk berbagai kombinasi parameter
'''learning_rates = np.round(np.arange(0.1, 1.1, 0.1), 2)  # Learning rate dari 0.1 hingga 1.0
weight_inits = np.round(np.arange(0.1, 1.1, 0.1), 2)  # Weight init dari 0.1 hingga 1.0
configs = range(1, 11)                    # Konfigurasi dari 3-1-3 hingga 3-10-3

# Loop kombinasi parameter untuk membuat banyak animasi
for lr in learning_rates:
    for wi in weight_inits:
        for config in configs:
            generate_animation(lr, wi, config)'''

weight_inits = 0.3
learning_rates = np.round(np.arange(0.2, 1.1, 0.1), 2)
configs = range(1, 11)

for lr in learning_rates:
    for config in configs:
        generate_animation(lr, weight_inits, config)


