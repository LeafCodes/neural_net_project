from fungsi_pengujian import variasi_neuron, variasi_neuron_test
from preprocessing_data import X, y
nn1, nn1_mse, nn1_accuracy, nn1_weight_total = variasi_neuron(X, y)
nn_vhl, nn_vhl_mse, nn_vhl_accuracy, nn_vhl_weight_total = variasi_neuron_test (X, y, variant_t='default')