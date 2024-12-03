import os
import numpy as np
import pandas as pd
import time

data = pd.read_csv(f'{os.getcwd()}\data\datalemari(-1,1).csv')

X = data.iloc[:,:2].values.T
y = data.iloc[:,2:5].values.T

#Normalisasi data
def minmax_scaling (x_t):
    min = np.min(x_t)
    max = np.max(x_t)
    return ((x_t-min)/(max-min))

X = minmax_scaling(data.iloc[:,:2].values.T)
y = data.iloc[:,2:5].values.T

def polynom_curve_fit(x_t, y_t, degree_t):
  coefficients = np.polyfit(x_t, y_t, degree_t)
  poly_function = np.poly1d(coefficients)
  x_fit = np.linspace(min(x_t), max(x_t), 100)
  y_fit = poly_function(x_fit)
  return x_fit, y_fit

def check_folder(folder_path):
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def save_data (neural_network_t, mse_epoch_t, accuracy_epoch_t, weight_total_epoch_t, folder_path_t):
  # Call check_folder with the directory path
  check_folder(folder_path_t)

  # Create the full file path for saving
  mse_epoch_file        = os.path.join(folder_path_t, 'mse_epoch.npy')
  mse_file              = os.path.join(folder_path_t, 'mse.npy')
  accuracy_epoch_file   = os.path.join(folder_path_t, 'accuracy_epoch.npy')
  accuracy_file         = os.path.join(folder_path_t, 'accuracy.npy')

  # Save the NumPy arrays to the file paths
  np.save(mse_epoch_file,       np.array(mse_epoch_t))
  np.save(mse_file,             np.array(neural_network_t.mse))
  np.save(accuracy_epoch_file,  np.array(accuracy_epoch_t))
  np.save(accuracy_file,        np.array(neural_network_t.accuracy))

  for j in range (neural_network_t.num_weight):
    # Create the full file path for saving
    weight_total_file         = os.path.join(folder_path_t, f'weight_total_{j}.npy')
    weight_total_epoch_file   = os.path.join(folder_path_t, f'weight_total_epoch_{j}.npy')

    # Save the NumPy arrays to the file paths
    np.save(weight_total_file,        np.array(neural_network_t.whole_weight[j]))
    np.save(weight_total_epoch_file,  np.array(weight_total_epoch_t[j]))

def running_average(new_value, current_average, n):
    """
    Menghitung rata-rata yang diperbarui setiap kali ada data baru.

    Args:
    new_value (float): Nilai baru yang ditambahkan ke dalam kumpulan data.
    current_average (float): Rata-rata sebelumnya dari semua data.
    n (int): Jumlah data sebelum nilai baru ditambahkan.

    Returns:
    float: Rata-rata yang diperbarui setelah nilai baru ditambahkan.
    """
    updated_average = (current_average * n + new_value) / (n + 1)
    return updated_average