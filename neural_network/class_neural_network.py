import numpy as np
import time
from preprocessing_data import running_average

class NeuralNetwork:
  def __init__(self, input_size, hidden_size, output_size, weight_init=0.5, beta=0.5, gamma=0.99, beta1=0.9, beta2=0.999):
    self.input_size = input_size
    self.hidden_size = np.array(hidden_size) if isinstance(hidden_size, list) else hidden_size
    self.output_size = output_size
    if hidden_size is None:
      self.num_layers = np.concatenate(([input_size], [output_size]))
    else:
      self.num_layers = np.concatenate(([input_size], hidden_size, [output_size]))
    self.num_weight = len(self.num_layers) - 1
    self.weight = {}
    self.whole_weight = {}
    self.v_t = {}
    self.G_t = {}
    self.beta = beta
    self.gamma = gamma
    self.beta1 = beta1
    self.beta2 = beta2
    self.accuracy = []
    self.mse = []
    self.t = 0
    self.backward_time_avg = 0
    self.no_bigger_mse = False
    self.no_d2J_dw2 = False
    self.no_bigger_grad = False

    for layer_index in range(self.num_weight):
      shape = (self.num_layers[layer_index+1], self.num_layers[layer_index]+1)
      self.weight[layer_index]            = self.initialize_weights(shape, weight_init)
      self.whole_weight[layer_index]      = np.zeros((1000000, *shape))
      self.whole_weight[layer_index][0]   = self.weight[layer_index]
      self.v_t[layer_index]               = np.zeros(shape)
      self.G_t[layer_index]               = np.zeros(shape)

  def initialize_weights(self, shape, method):
    if method == 'random':
      np.random.seed(42)
      return np.random.rand(*shape)
    elif method == 'HE':
      return np.random.normal(0, np.sqrt(2.0 / shape[1]), size=shape)
    elif method == 'Xavier':
      return np.random.normal(0, np.sqrt(2.0 / sum(shape)), size=shape)
    elif isinstance(method, (int, float)):
      return method * np.ones(shape)
    else:
      raise ValueError("weight_init harus 'random', 'HE', 'Xavier', atau angka.")

  def forward(self, X, activation):
    # Menambahkan bias pada bobot input
    self.input_with_bias  = np.vstack([np.ones((1, X.shape[1])), X])
    Y = {}
    self.F = {}

    for layer_index in range(self.num_weight):
      Y[layer_index]        = np.dot(self.weight[layer_index], self.input_with_bias if layer_index == 0 else self.F[layer_index-1])
      self.F[layer_index]   = self.activation_function(Y[layer_index], activation)
      if layer_index < self.num_weight - 1:
        self.F[layer_index]   = np.vstack([np.ones((1, self.F[layer_index].shape[1])), self.F[layer_index]])

    self.Y_out    = Y[self.num_weight-1]
    self.F_out    = self.F[self.num_weight-1]
    self.F_Step   = self.step_function(self.Y_out)

    return self.Y_out, self.F_out, self.F_Step

  def backward(self, X, y, learning_rate=0.1, optimizer='default', act_variants='Linear', method='per-element', stop_criterion = 'default', e = 1):
    backward_time   = time.time()
    break_all_loop  = False          # Break Condition
    self.forward(X, act_variants)   # Forward pass awal
    for layer_index in range(self.num_weight-1, -1, -1):    # Weight [layer_index] [][]
      if method == 'per-element':
        for j in range(self.num_layers[layer_index+1]):     # Weight [] [j][]
          for i in range(self.num_layers[layer_index]+1):     # Weight [] [][i]
            while True:
              self.update(layer_index, y, act_variants)     # Update gradient
              self.t+= 1

              if self.no_d2J_dw2 == True:
                self.d2J_dw2[layer_index]       = np.ones(self.weight[layer_index].shape)

              # Perhitungan akurasi dan MSE
              self.save_acc_mse(y)

              w_n                             = self.Gradien_Descent(optimizer, layer_index, j, i, learning_rate_t=learning_rate, update_mode=method)
              error                           = self.error(w_n, self.weight[layer_index][j, i])
              temp                            = self.weight[layer_index][j, i]
              self.weight[layer_index][j, i]  = w_n

              self.forward(X, act_variants)

              # Menyimpan nilai bobot baru ke variable whole_weight untuk dianalisis
              self.save_weight()

              if self.no_bigger_mse == True:
                if self.mse_func(y, self.F_out) >= self.mse[-1]:
                  self.weight[layer_index][j, i] = temp
                  self.whole_weight[layer_index][self.t]  = np.zeros(self.weight[layer_index].shape)
                  self.mse.pop(-1)
                  self.accuracy.pop(-1)
                  self.forward(X, act_variants)
                  self.t -= 1
                  break

                else :
                  self.backward_time_avg          = running_average(time.time()-backward_time, self.backward_time_avg, self.t-1)

              if stop_criterion == 'default':

                if self.accuracy_func(y, self.F_Step) == 100:   # Pelatihan berhenti jika mencapai 100%

                  self.save_acc_mse(y)
                  break_all_loop = True

                  break


              elif stop_criterion == 'nothing':
                pass

              else :
                raise ValueError("stop_criterion harus 'default' atau 'nothing'.")

              if error < e:
                break
            if break_all_loop:
              break
          if break_all_loop:
            break
        if break_all_loop:
          break

      elif method == 'per-layer':
        while True:
          self.update(layer_index, y, act_variants)   # Update gradient
          self.t += 1

          if self.no_d2J_dw2 == True:
            self.d2J_dw2[layer_index]       = np.ones(self.weight[layer_index].shape)

          # Perhitungan akurasi dan MSE
          self.save_acc_mse(y)

          # Update bobot
          w_n = self.Gradien_Descent(optimizer, layer_index, None, None, learning_rate_t=learning_rate, update_mode=method)
          error = np.sum(self.error(w_n, self.weight[layer_index]))
          temp = self.weight[layer_index]
          self.weight[layer_index] = w_n

          # Menyimpan bobot baru di variabel whole_weight
          self.save_weight ()

          # Forward pass ulang setelah pembaharuan bobot
          self.forward(X, act_variants)

          if self.no_bigger_mse == True:
            if self.mse_func(y, self.F_out) >= self.mse[-1]:
              self.weight[layer_index] = temp
              self.whole_weight[layer_index][self.t]  = np.zeros(self.weight[layer_index].shape)
              self.forward(X, act_variants)
              self.t -= 1
              break

            else :
              self.backward_time_avg          = running_average(time.time()-backward_time, self.backward_time_avg, self.t-1)


          if stop_criterion == 'default':

            if self.accuracy_func(y, self.F_Step) == 100: # Pelatihan berhenti jika akurasi mencapai 100%
              self.save_acc_mse(y)
              break_all_loop = True
              break

          elif stop_criterion == 'nothing':
            pass

          else :
            raise ValueError("stop_criterion harus 'default' atau 'nothing'.")

          if error < 0.1:
            break
        if break_all_loop:
          break

      else :
        raise ValueError("Method harus 'per-element' atau 'per-layer'.")



  def update(self, w_t, y, update_variants):
    m = self.input_with_bias.shape[1]
    self.dJ_dw = {}
    self.d2J_dw2 = {}
    dJ_dF = (-2 / m) * (y - self.F_out)


    # Gradient pertama
    for layer_index in range(self.num_weight-1, w_t - 1, -1):
      dF_dY = self.derivative_activation_function(self.F[layer_index], update_variants)
      dY_dF = self.weight[layer_index]

      if layer_index == self.num_weight-1:
        delta = dJ_dF * dF_dY
      else:
        delta = np.dot(self.weight[layer_index + 1][:,1:].T, delta) * dF_dY[1:,:]

      input_term = self.input_with_bias.T if layer_index == 0 else self.F[layer_index - 1].T
      self.dJ_dw[layer_index] = np.dot(delta, input_term)

    # Gradient kedua
    temp = np.zeros_like(self.weight[w_t])

    for layer_index in range(self.num_weight-1, w_t - 1, -1): #(2, 0, -1) 1
      b = dJ_dF
      #print('dJ_dF :', dJ_dF)
      #print('layer index :', layer_index)

      for k in range(self.num_weight-1, w_t - 1, -1): # (2, 0, -1) 2
        #print('k :', k)
        if k > layer_index:
          if k == self.num_weight-1:
            b = np.dot(self.weight[k][:, 1:].T, b * self.derivative_activation_function(self.F[k], update_variants))
          else:
            b = np.dot(self.weight[k][:, 1:].T, b * self.derivative_activation_function(self.F[k][1:, :], update_variants))
          #print('b k > layer_i :', b.shape)
        elif k == layer_index:
          if k == self.num_weight-1:
            b *= self.second_derivative_activation_function(self.F[k], update_variants)
          else:
            b *= self.second_derivative_activation_function(self.F[k][1:, :], update_variants)
          #print('b k = layer_i :', b.shape)
        else:
          if k == self.num_weight-1:
            b = np.dot(self.weight[k + 1][:, 1:].T ** 2, b) * self.derivative_activation_function(self.F[k], update_variants) ** 2 #[6, 16]
          else:
            b = np.dot(self.weight[k + 1][:, 1:].T ** 2, b) * self.derivative_activation_function(self.F[k][1:, :], update_variants) ** 2 #[6, 16]
         #print('b else :', b.shape)

      input_term = self.input_with_bias.T if w_t == 0 else self.F[w_t - 1].T
      #print('input term :', input_term.shape)
      temp += np.dot(b, input_term ** 2)

    a = (2 / m) * self.derivative_activation_function(self.F_out, update_variants) ** 2

    for j in range(self.num_weight-1, w_t, -1):
      a = np.dot(self.weight[j][:, 1:].T ** 2, a) * self.derivative_activation_function(self.F[j - 1][1:, :], update_variants) ** 2

    temp += np.dot(a, input_term ** 2)
    self.d2J_dw2[w_t] = temp

    return

  def Gradien_Descent(self, variant, w, j=None, i=None, learning_rate_t=0.01, update_mode='per-element'):
    # Pilih variabel berdasarkan update_mode
    if update_mode == 'per-element':
        a, b, c, d, e = (
            self.dJ_dw[w][j, i],
            self.d2J_dw2[w][j, i],
            self.weight[w][j, i],
            self.v_t[w][j, i],
            self.G_t[w][j, i]
        )
    elif update_mode == 'per-layer':
        a, b, c, d, e = (
            self.dJ_dw[w],
            self.d2J_dw2[w],
            self.weight[w],
            self.v_t[w],
            self.G_t[w]
        )
    else:
        raise ValueError("update_mode harus 'per-element' atau 'per-layer'")

    # Implementasi variant
    if variant == 'default':
        temp = a / b
    elif variant == 'momentum':
        d = self.beta * d + (1 - self.beta) * a
        temp = d / b
    elif variant == 'adagrad':
        e += a ** 2
        temp = (a / b) / np.sqrt(e + 1e-8)
    elif variant == 'rmsprop':
        e = self.beta2 * e + (1 - self.beta2) * a**2
        temp = a / (b * (np.sqrt(e / (1 - self.beta2)) + 1e-8))
    elif variant == 'adam':
        e = self.beta2 * e + (1 - self.beta2) * a**2
        d = self.beta1 * d + (1 - self.beta1) * a
        temp = (d / (1 - self.beta1)) / (np.sqrt(e / (1 - self.beta2)) + 1e-8) / b
    else:
        raise ValueError("Variant tidak ditemukan")

    # Cegah gradien terlalu besar
    if self.no_bigger_grad and np.abs(temp) > 0:
        temp = self.ubah_ke_angka_desimal(temp)

    # Perbarui bobot
    return c - learning_rate_t * temp

  def ubah_ke_angka_desimal(self, nilai):
    b = np.abs(np.min(nilai))
    a = str(abs(int(b)))
    if len(a) == 1 :
      return(nilai)
    else :
      return nilai / (10 ** len(a))

  def activation_function(self, x, variant):
    variants = {
        'Linear': x,
        'sigmoid': 1 / (1 + np.exp(-x)),
        'tanh': np.tanh(x),
        'relu': np.maximum(0, x),
        'leaky_relu': np.where(x > 0, x, 0.01 * x),
        'elu': np.where(x > 0, x, np.exp(x)-1),
        'softmax': np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
    }
    return variants.get(variant, x)

  def derivative_activation_function(self, x, variant):
    variants = {
        'Linear': np.ones_like(x),
        'sigmoid': x * (1 - x),
        'tanh': 1 - x**2,
        'relu': np.where(x > 0, 1, 0),
        'leaky_relu': np.where(x > 0, 1, 0.01),
        'elu': np.where(x > 0, 1, np.exp(x)),
        'softmax': x * (1 - x)  # Simplification for single output softmax
    }
    return variants.get(variant, x)

  def second_derivative_activation_function(self, x, variant):
    variants = {
        'Linear': np.zeros_like(x),
        'sigmoid': x * (1 - x) * (1 - 2 * x),
        'tanh': -2 * x * (1 - x**2),
        'relu': np.zeros_like(x),
        'leaky_relu': np.where(x > 0, 0, 0.01),
        'elu': np.where(x > 0, 0, np.exp(x)),
        'softmax': x * (1 - x) * (1 - 2 * x)  # Simplification for single output softmax
    }
    return variants.get(variant, np.zeros_like(x))

  def save_acc_mse (self, y_t):
    acc = self.accuracy_func(y_t, self.F_Step)
    mse = self.mse_func(y_t, self.F_out)
    self.accuracy.append(acc)
    self.mse.append(mse)

    return
  def save_weight (self):
    for k in range(self.num_weight):
      self.whole_weight[k][self.t]  = self.weight[k]

  def accuracy_func(self, a_t, b_t):
    return np.mean(np.all(a_t == b_t, axis=0)) * 100

  def step_function(self, x, threshold=0):
    return np.where(x >= threshold, 1, -1)

  def error(self, a, b):
    return np.abs(1-(a/b))

  def accuracy_func(self, a_t, b_t):
    return np.mean(np.all(a_t == b_t, axis=0)) * 100

  def mse_func(self, d_t, y_t):
    return np.mean(np.square(d_t - y_t))

  def he_initializer(self, input_size, output_size):
    return np.random.normal(0, np.sqrt(2.0 / input_size), size=(input_size, output_size))

  def xavier_initializer(self, input_size, output_size):
    return np.random.normal(0, np.sqrt(2.0 / (input_size + output_size)), size=(input_size, output_size))

  def train(self, X, y, epochs, l_r, variant_optimasi, fungsi_aktivasi, update_method_t='per_element', stop_criterion_t='default', er = 0.1):
    train_time = time.time()
    weight_epoch = {}

    for layer_index in range (self.num_weight):
      weight_epoch[layer_index]       = np.zeros((epochs+1, *self.weight[layer_index].shape))
      weight_epoch[layer_index][0]    = self.whole_weight[layer_index][0]

    self.backward(X, y , learning_rate=l_r, optimizer=variant_optimasi, act_variants=fungsi_aktivasi, method=update_method_t, stop_criterion=stop_criterion_t, e=er)
    mse , accuracy = [], []

    for layer_index in range (self.num_weight):
      weight_epoch[layer_index][1]    = self.weight[layer_index]

    for epoch in range (2, epochs+1):
      old_mse = self.mse[-1]
      mse.append(old_mse)
      accuracy.append(self.accuracy[-1])
      self.backward(X, y , learning_rate=l_r, optimizer=variant_optimasi, act_variants=fungsi_aktivasi, method=update_method_t, stop_criterion=stop_criterion_t, e=er)
      self.forward(X, fungsi_aktivasi)
      new_mse = self.mse[-1]

      for layer_index in range (self.num_weight):
        weight_epoch[layer_index][epoch] = self.weight[layer_index]

      if stop_criterion_t == 'default':

        if new_mse == old_mse :
          print(f'Epoch {epoch}, MSE: {mse[-1]}, Accuracy: {accuracy[-1]}%')
          print('Training stopped because no improvement in loss')

          for layer_index in range(self.num_weight):
            weight_epoch[layer_index]       = np.delete(weight_epoch[layer_index], np.s_[epoch + 1:], axis=0)
            self.whole_weight[layer_index]  = np.delete(self.whole_weight[layer_index], np.s_[self.t + 1:], axis=0)
          break

        if epoch % 100 == 0 or self.accuracy[-1] == 100:
          print(f'Epoch {epoch}, MSE: {new_mse}, Accuracy: {self.accuracy[-1]}%, Backward time average: {self.backward_time_avg}')

          if self.accuracy[-1] == 100:
            print('waktu train :',time.time()-train_time)
            print('Target Accuracy Reached')
            mse.append(new_mse)
            accuracy.append(self.accuracy[-1])

            for layer_index in range(self.num_weight):
              weight_epoch[layer_index]     = np.delete(weight_epoch[layer_index], np.s_[epoch + 1:], axis=0)
              self.whole_weight[layer_index]  = np.delete(self.whole_weight[layer_index], np.s_[self.t + 1:], axis=0)
            break

      elif stop_criterion_t == 'nothing':

        if epoch % 100 == 0:
          print(f'Epoch {epoch}, MSE: {new_mse}, Accuracy: {self.accuracy[-1]}%')
      else :
        raise ValueError("stop_criterion harus 'default' atau 'nothing'.")
    return mse, accuracy, weight_epoch