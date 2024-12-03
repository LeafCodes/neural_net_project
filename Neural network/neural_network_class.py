import numpy as np
import time
from running_average import running_average
class NeuralNetwork:
  def __init__(self, input_size, hidden_size, output_size, weight_init=0.5, beta=0.5, gamma=0.99, beta1=0.9, beta2=0.999):
    self.input_size = input_size
    self.hidden_size = np.array(hidden_size) if isinstance(hidden_size, list) else hidden_size
    self.output_size = output_size
    self.num_layers = np.concatenate(([input_size+1], hidden_size, [output_size]))
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

    for w in range(self.num_weight):
      shape = (self.num_layers[w+1], self.num_layers[w])
      self.weight[w] = self.initialize_weights(shape, weight_init)
      self.whole_weight[w] = np.zeros((1000000, *shape))
      self.whole_weight[w][0] = self.weight[w]
      self.v_t[w] = np.zeros(shape)
      self.G_t[w] = np.zeros(shape)

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
    self.X  = np.vstack([np.ones((1, X.shape[1])), X])
    self.Y = {}
    self.F = {}

    for w in range(self.num_weight):
      self.Y[w] = np.dot(self.weight[w], self.X if w == 0 else self.F[w-1])
      self.F[w] = self.activation_function(self.Y[w], activation)

    self.Y_out  = self.Y[self.num_weight-1]
    self.F_out  = self.F[self.num_weight-1]
    self.F_Step = self.step_function(self.Y_out)

    return self.Y_out, self.F_out, self.F_Step

  def backward(self, X, y, learning_rate=0.1, Opt_variant='default', act_variants='Linear', method='per-element', stop_condition = 'default'):
    backward_time = time.time()
    self.forward(X, act_variants)   # Forward pass awal
    break_all_loop = False          # Break Condition

    for w in range(self.num_weight-1, -1, -1):
      if method == 'per-element':
        for j in range(self.num_layers[w+1]):
          for i in range(self.num_layers[w]):
            while True:
              self.update(w, y, act_variants)
              self.t += 1

              # Perhitungan akurasi dan MSE
              acc = self.accuracy_func(y, self.F_Step)
              mse = self.mse_func(y, self.F_out)
              self.accuracy.append(acc)
              self.mse.append(mse)

              w_n = self.Gradien_Descent(Opt_variant, w, j, i, learning_rate_t=learning_rate, update_mode=method)
              error = self.error(w_n, self.weight[w][j, i])
              self.weight[w][j, i] = w_n

              for k in range(self.num_weight):
                self.whole_weight[k][self.t] = self.weight[k]

              # Forward pass ulang setelah pembaharuan bobot
              self.forward(X, act_variants)
              
              self.backward_time_avg = running_average(time.time()-backward_time, self.backward_time_avg, self.t-1)
              
              if stop_condition == 'default':
                if self.accuracy_func(y, self.F_Step) == 100:
                  acc = self.accuracy_func(y, self.F_Step)
                  mse = self.mse_func(y, self.F_out)
                  self.accuracy.append(acc)
                  self.mse.append(mse)
                  break_all_loop = True
                  
                  break
              elif stop_condition == 'nothing':
                pass
              else :
                raise ValueError("stop_condition harus 'default' atau 'nothing'.")

              if error < 0.1:
                break
            if break_all_loop:
              break
          if break_all_loop:
            break
        if break_all_loop:
          break

      elif method == 'per-layer':
        while True:
          self.update(w, y, act_variants)
          self.t += 1

          # Perhitungan akurasi dan MSE
          acc = self.accuracy_func(y, self.F_Step)
          mse = self.mse_func(y, self.F_out)
          self.accuracy.append(acc)
          self.mse.append(mse)

          w_n = self.Gradien_Descent(Opt_variant, w, None, None, learning_rate_t=learning_rate, update_mode=method)
          error = np.sum(self.error(w_n, self.weight[w]))
          self.weight[w] = w_n

          for k in range(self.num_weight):
            self.whole_weight[k][self.t] = self.weight[k]

          # Forward pass ulang setelah pembaharuan bobot
          self.forward(X, act_variants)

          if stop_condition == 'default':
            if self.accuracy_func(y, self.F_Step) == 100:
              acc = self.accuracy_func(y, self.F_Step)
              mse = self.mse_func(y, self.F_out)
              self.accuracy.append(acc)
              self.mse.append(mse)
              break_all_loop = True
              break

          elif stop_condition == 'nothing':
            pass

          else :
            raise ValueError("stop_condition harus 'default' atau 'nothing'.")

          if error < 0.1:
            break
        if break_all_loop:
          break


  def update(self, w_t, y, update_variants):
    m = self.X.shape[1]
    self.dJ_dw = {}
    self.d2J_dw2 = {}
    dJ_dF = (-2 / m) * (y - self.F_out)

    for w in range(len(self.hidden_size), w_t - 1, -1):
      dF_dY = self.derivative_activation_function(self.F[w], update_variants)
      dY_dF = self.weight[w]

      if w == len(self.hidden_size):
        delta = dJ_dF * dF_dY
      else:
        delta = np.dot(self.weight[w + 1].T, delta) * dF_dY

      input_term = self.X.T if w == 0 else self.F[w - 1].T
      self.dJ_dw[w] = np.dot(delta, input_term)

    self.d2J_dw2[w_t] = np.zeros_like(self.weight[w_t])

    for s in range(len(self.hidden_size), w_t - 1, -1):
      b = dJ_dF

      for k in range(len(self.hidden_size), w_t - 1, -1):
        if k > s:
          b = np.dot(self.weight[k].T, b * self.derivative_activation_function(self.F[k], update_variants))
        elif k == s:
          b *= self.second_derivative_activation_function(self.F[s], update_variants)
        else:
          b = np.dot(self.weight[k + 1].T ** 2, b) * self.derivative_activation_function(self.F[k], update_variants) ** 2

      input_term = self.X.T if w_t == 0 else self.F[w_t - 1].T
      self.d2J_dw2[w_t] += np.dot(b, input_term ** 2)

    a = (2 / m) * self.derivative_activation_function(self.F_out, update_variants) ** 2

    for j in range(len(self.hidden_size), w_t, -1):
      a = np.dot(self.weight[j].T ** 2, a) * self.derivative_activation_function(self.F[j - 1], update_variants) ** 2

    self.d2J_dw2[w_t] += np.dot(a, input_term ** 2)

    return

  def Gradien_Descent(self, variant, w, j, i, learning_rate_t, update_mode='per-element'):
    if update_mode == 'per-element' :
      if variant == 'default' :
        return self.weight[w][j, i] - learning_rate_t * (self.dJ_dw[w][j, i]/self.d2J_dw2[w][j, i])

      elif variant == 'momentum' :
        self.v_t[w][j, i]  = (self.beta * self.v_t[w][j, i])  + ((1-self.beta) * self.dJ_dw[w][j, i])
        return self.weight[w][j, i] - learning_rate_t * (self.v_t[w][j, i]/self.d2J_dw2[w][j, i])

      elif variant == 'adagrad' :
        self.G_t[w][j, i] += self.dJ_dw[w][j, i] ** 2
        return self.weight[w][j, i] - (learning_rate_t * (self.dJ_dw[w][j, i]/self.d2J_dw2[w][j, i])) / np.sqrt(self.G_t[w][j, i] + 1e-8)

      elif variant == 'rmsprop' :
        self.G_t[w][j, i] = self.G_t[w][j, i] * self.beta2 + (1-self.beta2) * self.dJ_dw[w][j, i]**2
        return self.weight[w][j, i] - (learning_rate_t / (np.sqrt(self.G_t[w][j, i]/(1-self.beta2)) + 1e-8)) * (self.dJ_dw[w][j, i]/self.d2J_dw2[w][j, i])

      elif variant == 'adam' :
        self.G_t[w][j, i] = self.G_t[w][j, i] * self.beta2 + (1-self.beta2) * self.dJ_dw[w][j, i]**2
        self.v_t[w][j, i] = self.v_t[w][j, i] * self.beta1 + (1-self.beta1) * self.dJ_dw[w][j, i]
        return self.weight[w][j, i] - (learning_rate_t/(np.sqrt(self.G_t[w][j, i]/(1-self.beta2)) + 1e-8)) * (self.v_t[w][j, i]/(1-self.beta1))

      else :
        print('Variant tidak ditemukan')
        return

    elif update_mode == 'per-layer' :
      if variant == 'default' :
        return self.weight[w] - learning_rate_t * (self.dJ_dw[w]/self.d2J_dw2[w])

      elif variant == 'momentum' :
        self.v_t[w]  = (self.beta * self.v_t[w])  + ((1-self.beta) * self.dJ_dw[w])
        return self.weight[w] - (learning_rate_t * (self.v_t[w]/self.d2J_dw2[w]))

      elif variant == 'adagrad' :
        self.G_t[w] += self.dJ_dw[w] ** 2
        return self.weight[w] - (learning_rate_t * (self.dJ_dw[w]/self.d2J_dw2[w]))/ np.sqrt(self.G_t[w] + 1e-8)

      elif variant == 'rmsprop' :
        self.G_t[w] = self.G_t[w] * self.beta2 + (1-self.beta2) * self.dJ_dw[w]**2
        return self.weight[w] - ((learning_rate_t / (np.sqrt(self.G_t[w]/(1-self.beta2)) + 1e-8)) * (self.dJ_dw[w] / self.d2J_dw2[w]))

      elif variant == 'adam' :
        self.G_t[w] = self.G_t[w] * self.beta2 + (1-self.beta2) * self.dJ_dw[w]**2
        self.v_t[w] = self.v_t[w] * self.beta1 + (1-self.beta1) * self.dJ_dw[w]
        return self.weight[w] - (learning_rate_t / (np.sqrt(self.G_t[w]/(1-self.beta2)) + 1e-8)) * (self.v_t[w] / (1-self.beta1))

      else :
        print('Variant tidak ditemukan')
        return

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

  def train(self, X, y, epochs, l_r, variant_optimasi, fungsi_aktivasi, method_t='per_element', stop_condition_t='default'):
    train_time = time.time()
    weight_epoch = {}
    for w in range (self.num_weight):
      weight_epoch[w] = np.zeros((10000, *self.weight[w].shape))
      weight_epoch[w][0] = self.whole_weight[w][0]
    
    self.backward(X, y , learning_rate=l_r, Opt_variant=variant_optimasi, act_variants=fungsi_aktivasi, method=method_t, stop_condition=stop_condition_t)
    mse , accuracy = [], []
    for w in range (self.num_weight):
      weight_epoch[w][1] = self.weight[w]
    for epoch in range (2, epochs+1):
      old_mse = self.mse[-1]
      mse.append(old_mse)
      accuracy.append(self.accuracy[-1])
      self.backward(X, y , learning_rate=l_r, Opt_variant=variant_optimasi, act_variants=fungsi_aktivasi, method=method_t, stop_condition=stop_condition_t)
      self.forward(X, fungsi_aktivasi)
      new_mse = self.mse[-1]
      for w in range (self.num_weight):
        weight_epoch[w][epoch] = self.weight[w]

      if stop_condition_t == 'default':
        if new_mse == old_mse :
          print(f'Epoch {epoch}, MSE: {mse[-1]}, Accuracy: {accuracy[-1]}%')
          print('Training stopped because no improvement in loss')
          break
        if epoch % 100 == 0 or self.accuracy[-1] == 100:
          print(f'Epoch {epoch}, MSE: {new_mse}, Accuracy: {self.accuracy[-1]}%')
          print('Backward time average:', self.backward_time_avg)
          if self.accuracy[-1] == 100:
            print('waktu train :',time.time()-train_time)
            print('Target Accuracy Reached')
            mse.append(new_mse)
            accuracy.append(self.accuracy[-1])
            break
      elif stop_condition_t == 'nothing':
        if epoch % 100 == 0:
          print(f'Epoch {epoch}, MSE: {new_mse}, Accuracy: {self.accuracy[-1]}%')
      else :
        raise ValueError("stop_condition harus 'default' atau 'nothing'.")
    return mse, accuracy, weight_epoch