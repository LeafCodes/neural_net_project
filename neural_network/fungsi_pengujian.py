import time
from class_neural_network import NeuralNetwork
import os

weight_awal = 0.3             # dapat berupa angka, 'random', 'HE', 'Xavier'
learning_rate = 0.1
epoch = 10000
optimasi = 'default'            # 'default', 'momentum', 'adagrad', 'rmsp', atau 'adam'
aktivasi = 'Linear'             # 'Linear', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', atau 'softmax'
kondisi_stop = 'default'        # 'default' or 'nothing'
cara_pelatihan = 'per-element'  # 'per-element' or 'per-layer'
no_bigger_grad = False
no_bigger_mse = True
no_d2J_dw2 = False

def normal_test (X, y, variant_t, hidden_layer_t) :
  first_time = time.time()
  nn = NeuralNetwork(2, hidden_layer_t, 3, weight_init=weight_awal)
  nn.no_bigger_grad = no_bigger_grad
  nn.no_bigger_mse = no_bigger_mse
  nn.no_d2J_dw2 = no_d2J_dw2
  nn_mse = []
  nn_accuracy = []
  nn_weight_total = []
  nn_mse, nn_accuracy, nn_weight_total = nn.train(X, y, epoch, learning_rate, variant_t, aktivasi, cara_pelatihan, kondisi_stop)
  last_time = time.time()
  print(f'waktu yang dibutuhkan : {last_time-first_time}')

  return nn, nn_mse, nn_accuracy, nn_weight_total

def variasi_neuron (X, y, variant_t=optimasi):
  first_time = time.time()
  nn = {}
  nn_mse = {}
  nn_accuracy = {}
  nn_weight_total = {}
  for n in range (1, 11):
    print(f'Hidden Neuron : {n}')
    nn[n], nn_mse[n], nn_accuracy[n], nn_weight_total[n] = normal_test(X, y, variant_t, [n])

  return nn, nn_mse, nn_accuracy, nn_weight_total

def variasi_neuron_test (X, y, variant_t=optimasi):
  nn_vhl = {}
  nn_vhl_mse = {}
  nn_vhl_accuracy = {}
  nn_vhl_weight_total = {}
  for m in range (2, 7):
    start_hidden_layers = 1
    last_hidden_layers = 4
    hidden_neuron = []
    for n in range (start_hidden_layers, last_hidden_layers+1):
      print(f'number of hidden layer : {n}')
      print(f'number of neuron each hidden layer : {m}')
      hidden_neuron.append(m)
      nn_vhl[f'{m}{n}'], nn_vhl_mse[f'{m}{n}'], nn_vhl_accuracy[f'{m}{n}'], nn_vhl_weight_total[f'{m}{n}'] = normal_test(X, y, variant_t, hidden_neuron)

  return nn_vhl, nn_vhl_mse, nn_vhl_accuracy, nn_vhl_weight_total

def check_folder(folder_path):
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)