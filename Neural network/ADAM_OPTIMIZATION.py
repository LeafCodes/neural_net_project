import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('/Users/ASUS/Downloads/data')
data = pd.read_csv('datalemari.csv', sep=',', header=0)
data.head()

# Define the deep learning model with a single hidden layer
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize the weights and biases
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
        
        # Initialize the Adam parameters
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
        # Forward pass through the network
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        
        # Backpropagation
        delta2 = self.a2 - y
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0) / m
        
        delta1 = np.dot(delta2, self.W2.T) * (1 - np.power(self.a1, 2))
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0) / m
        
        # Adam update for weights and biases
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

# Generate sample data
X = data.iloc[:,1:3].values
y = data.iloc[:,3:6].values

# Initialize the neural network
input_size = X.shape[1]
hidden_size = 4
output_size = y.shape[1]
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Training parameters
epochs = 10000
learning_rate = 0.1

# Training loop
for epoch in range(epochs):
    # Forward pass
    y_pred = nn.forward(X)
    
    # Backward pass and weight update
    nn.backward(X, y, learning_rate)
    
    # Print the loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean((y_pred - y) ** 2)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
print("Predictions :\n",np.around(y_pred, 3))