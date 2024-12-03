import numpy as np
import pandas as pd
from timeit import default_timer as timer
import os

os.chdir('/Users/ASUS/Downloads/data')
data = pd.read_csv('datalemari.csv', sep=',', header=0)
data.head()

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights = {}
        self.biases = {}
        self.m_W = {}
        self.v_W = {}
        self.m_b = {}
        self.v_b = {}
        self.dL_dW = {}
        self.dL_db = {}
        self.beta1 = 0.9 
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0

        # Initialize weights and biases
        self.weights['W0'] = np.random.rand(input_size, hidden_size[0])
        self.biases['b0'] = np.zeros((1, hidden_size[0]))
        self.m_W['0'] = np.zeros_like(self.weights['W0'])
        self.v_W['0'] = np.zeros_like(self.weights['W0'])
        self.m_b['0'] = np.zeros_like(self.biases['b0'])
        self.v_b['0'] = np.zeros_like(self.biases['b0'])

        for i in range(1, len(hidden_size)):
            self.weights[f'W{i}'] = np.random.rand(hidden_size[i-1], hidden_size[i])
            self.biases[f'b{i}'] = np.zeros((1, hidden_size[i]))
            self.m_W[f'{i}'] = np.zeros_like(self.weights[f'W{i}'])
            self.v_W[f'{i}'] = np.zeros_like(self.weights[f'W{i}'])
            self.m_b[f'{i}'] = np.zeros_like(self.biases[f'b{i}'])
            self.v_b[f'{i}'] = np.zeros_like(self.biases[f'b{i}'])

        self.weights[f'W{len(hidden_size)}'] = np.random.rand(hidden_size[-1], output_size)
        self.biases[f'b{len(hidden_size)}'] = np.zeros((1, output_size))
        self.m_W[f'{len(hidden_size)}'] = np.zeros_like(self.weights[f'W{len(hidden_size)}'])
        self.v_W[f'{len(hidden_size)}'] = np.zeros_like(self.weights[f'W{len(hidden_size)}'])
        self.m_b[f'{len(hidden_size)}'] = np.zeros_like(self.biases[f'b{len(hidden_size)}'])
        self.v_b[f'{len(hidden_size)}'] = np.zeros_like(self.biases[f'b{len(hidden_size)}'])

        self.layers = len(hidden_size) + 1

    def forward(self, X):
        self.activations = {'a0': X}

        for i in range(1, self.layers+1):
            z = np.dot(self.activations[f'a{i-1}'], self.weights[f'W{i-1}']) + self.biases[f'b{i-1}']
            if i == self.layers:
                self.activations[f'a{i}'] = self.sigmoid(z)
            else:
                self.activations[f'a{i}'] = self.Leaky_ReLU(z)

        return self.activations[f'a{self.layers}']
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]  # Number of data points

        # Backpropagation for output layer
        dL_da = 2*(self.activations[f'a{self.layers}'] - y) # Partial derivative of loss with respect to output activations
        da_dz = self.sigmoid_derivative(self.activations[f'a{self.layers}'])  # Partial derivative of output activations with respect to weighted sum

        dz_dW = self.activations[f'a{self.layers-1}'].T  # Partial derivative of weighted sum with respect to weights
        dz_db = np.ones((1, m))  # Partial derivative of weighted sum with respect to biases

        self.dL_dW[f'{self.layers-1}'] = np.dot(dz_dW, dL_da * da_dz)  # Partial derivative of loss with respect to weights
        self.dL_db[f'{self.layers-1}'] = np.dot(dz_db, dL_da * da_dz)  # Partial derivative of loss with respect to biases
        #print(f'dL_dw {self.layers-1} : \n', self.dL_dW[f'{self.layers-1}'])
        #print(f'Weight {self.layers-1}: \n', self.weights[f'W{self.layers-1}'])

        # Backpropagation for hidden layers
        for i in range(self.layers-2, -1, -1):
            dL_da = np.dot(dL_da * da_dz, self.weights[f'W{i+1}'].T)  # Partial derivative of loss with respect to output activations
            da_dz = self.derivative_Leaky_ReLU(self.activations[f'a{i+1}'])  # Partial derivative of output activations with respect to weighted sum
            
            dz_dW = self.activations[f'a{i}'].T  # Partial derivative of weighted sum with respect to weights
            dz_db = np.ones((1, m))  # Partial derivative of weighted sum with respect to biases

            self.dL_dW[f'{i}'] = np.dot(dz_dW, dL_da * da_dz)  # Partial derivative of loss with respect to weights
            self.dL_db[f'{i}'] = np.dot(dz_db, dL_da * da_dz)  # Partial derivative of loss with respect to biases
            #print(f'dL_dw {i} : \n', self.dL_dW[f'{i}'])
            #print(f'Weight {i}: \n', self.weights[f'W{i}'])

    
    def update_parameters(self, learning_rate):
        self.t += 1
        for i in range(self.layers):

            self.m_W[f'{i}'] = self.beta1 * self.m_W[f'{i}'] + (1 - self.beta1) * self.dL_dW[f'{i}']
            self.v_W[f'{i}'] = self.beta2 * self.v_W[f'{i}'] + (1 - self.beta2) * (self.dL_dW[f'{i}'] ** 2)
            m_W_hat = self.m_W[f'{i}'] / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W[f'{i}'] / (1 - self.beta2 ** self.t)

            self.m_b[f'{i}'] = self.beta1 * self.m_b[f'{i}'] + (1 - self.beta1) * self.dL_db[f'{i}']
            self.v_b[f'{i}'] = self.beta2 * self.v_b[f'{i}'] + (1 - self.beta2) * (self.dL_db[f'{i}'] ** 2)
            m_b_hat = self.m_b[f'{i}'] / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b[f'{i}'] / (1 - self.beta2 ** self.t)

            self.weights[f'W{i}'] -= learning_rate * (m_W_hat / (np.sqrt(v_W_hat) + self.eps))
            self.biases[f'b{i}'] -= learning_rate * (m_b_hat / (np.sqrt(v_b_hat) + self.eps))
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
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

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def train(self, X, y, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            # Forward propagation
            output = self.forward(X)

            # Backward propagation
            self.backward(X, y, learning_rate)

            # Update parameters
            self.update_parameters(learning_rate)

            # Calculate and print loss
            if epoch % 100 == 0:
                loss = np.mean(np.square(output-y))
                print(f"Epoch {epoch}/{num_epochs}, Loss: {loss}")


# Example usage
X = data.iloc[:,1:3].values
y = data.iloc[:,3:6].values

input_size = X.shape[1]
hidden_size = [4, 4, 4]
output_size = y.shape[1]

nn = NeuralNetwork(input_size, hidden_size, output_size)
Start = timer()
nn.train(X, y, learning_rate=0.01, num_epochs=10000)
print(nn.forward(X))
end = timer()
print("Waktu Komputasi :",end-Start)

