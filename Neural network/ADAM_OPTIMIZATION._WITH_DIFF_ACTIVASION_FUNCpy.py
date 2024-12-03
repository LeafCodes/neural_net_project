import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer

os.chdir('/Users/ASUS/Downloads/data')
data = pd.read_csv('datalemari.csv', sep=',', header=0)
data.head()

# Membuat class Neural Network
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size                                #(jumlah neuron pada input layer)
        self.hidden_size = hidden_size                              #(jumlah layer dan neuron pada masing2 layers)
        self.output_size = output_size                              #(jumlah neuron pada output layer)
        
        # Nilai Awal Dari Weight dan Bias
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
        
        # Nilai awal momentum dan hyperparameter dari optimasi adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        self.m_W2 = np.zeros_like(self.W2)
        self.v_W2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)
        self.t = 0
        
    def forward(self, X):
        # fungsi forward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.Leaky_ReLU(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]                                                      #jumlah training data
        
        # Backpropagation
        delta2 = self.a2 - y
        dW2 = np.dot(self.a1.T, delta2) / m                                 #gradien dari weight hidden - output
        db2 = np.sum(delta2, axis=0) / m                                    #gradien dari bias hidden - output
        
        delta1 = np.dot(delta2, self.W2.T) * self.derivative_Leaky_ReLU(self.z1)
        dW1 = np.dot(X.T, delta1) / m                                       #gradien dari weight input - hidden
        db1 = np.sum(delta1, axis=0) / m                                    #gradien dari bias input - hidden
        
        # update weight dan bias
        self.t += 1
        
        self.m_W1 = self.beta1 * self.m_W1 + (1 - self.beta1) * dW1
        self.v_W1 = self.beta2 * self.v_W1 + (1 - self.beta2) * (dW1 ** 2)
        m_hat_W1 = self.m_W1 / (1 - self.beta1 ** self.t)
        v_hat_W1 = self.v_W1 / (1 - self.beta2 ** self.t)
        self.W1 -= learning_rate * m_hat_W1 / (np.sqrt(v_hat_W1) + self.eps)
        
        self.m_b1 = self.beta1 * self.m_b1 + (1 - self.beta1) * db1
        self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * (db1 ** 2)
        m_hat_b1 = self.m_b1 / (1 - self.beta1 ** self.t)
        v_hat_b1 = self.v_b1 / (1 - self.beta2 ** self.t)
        self.b1 -= learning_rate * m_hat_b1 / (np.sqrt(v_hat_b1) + self.eps)
        
        self.m_W2 = self.beta1 * self.m_W2 + (1 - self.beta1) * dW2
        self.v_W2 = self.beta2 * self.v_W2 + (1 - self.beta2) * (dW2 ** 2)
        m_hat_W2 = self.m_W2 / (1 - self.beta1 ** self.t)
        v_hat_W2 = self.v_W2 / (1 - self.beta2 ** self.t)
        self.W2 -= learning_rate * m_hat_W2 / (np.sqrt(v_hat_W2) + self.eps)
        
        self.m_b2 = self.beta1 * self.m_b2 + (1 - self.beta1) * db2
        self.v_b2 = self.beta2 * self.v_b2 + (1 - self.beta2) * (db2 ** 2)
        m_hat_b2 = self.m_b2 / (1 - self.beta1 ** self.t)
        v_hat_b2 = self.v_b2 / (1 - self.beta2 ** self.t)
        self.b2 -= learning_rate * m_hat_b2 / (np.sqrt(v_hat_b2) + self.eps)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative_sigmoid (self, x):
        return (1 - self.sigmoid(self, x))
    
    def ReLU(self, x):
        data = x > 0
        result = x * data
        return result
    
    def derivative_ReLU(self, x):
        data = x > 0
        result = data * 1
        return result

    def Leaky_ReLU(self, x):
        data_1 = x > 0
        data_2 = x < 0
        result_1 = x * data_1
        result_2 = x * data_2 * 0.01
        result = result_1 + result_2
        return result
    
    def derivative_Leaky_ReLU(self, x):
        data_1 = x > 0
        data_2 = x < 0
        result_1 = 1 * data_1
        result_2 = 0.01 * data_2 * 0.01
        result = result_1 + result_2
        return result
# Generate data
X = data.iloc[:,1:3].values
y = data.iloc[:,3:6].values

# arsitektur dari neural network
input_size = X.shape[1]
hidden_size = 4
output_size = y.shape[1]
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Training parameters
epochs = 1000
learning_rate = 0.1
Start = timer()
# Training loop
for epoch in range(epochs):
    # Forward pass
    y_pred = nn.forward(X)
    
    # Backward pass and weight update
    nn.backward(X, y, learning_rate)
    
    # Print loss tiap 1000 epochs
    if epoch % 100 == 0:
        loss = np.mean((y_pred - y) ** 2)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
end = timer()
print("Waktu Komputasi :",end-Start)
print("Predictions :\n",np.around(y_pred, 3))