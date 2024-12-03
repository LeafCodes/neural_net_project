import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

os.chdir('/Users/ASUS/Downloads/skripsi/data/1 hl 2 neuron/run 7')
# Membaca file CSV tanpa menganggap baris pertama sebagai header
df = pd.read_csv('nilai f0.csv', header=None)

a = df.values[::2]
b = df.values[1::2]

# Fungsi untuk menggambar plot pada setiap iterasi
def update(frame):
    plt.clf()  # Hapus gambar sebelumnya
    x = a[frame, :]
    y = b[frame, :]

    # Menentukan warna untuk setiap kolom
    colors = ['r' if i < 5 else 'b' if i < 12 else 'g' for i in range(16)]

    plt.scatter(x, y, c=colors)
    plt.xlabel('Sumbu X')
    plt.ylabel('Sumbu Y')
    plt.title(f'F0 Epoch ke-{frame}')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid()

# Inisialisasi plot
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=a.shape[0], repeat=False, interval=10)

plt.show()
# Tampilkan animasi
