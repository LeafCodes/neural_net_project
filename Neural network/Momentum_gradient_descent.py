import numpy as np

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
        
        # Initialize the momentum terms
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        
    def forward(self, X):
        # Forward pass through the network
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate, momentum):
        m = X.shape[0]
        
        # Backpropagation
        delta2 = self.a2 - y
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0) / m
        
        delta1 = np.dot(delta2, self.W2.T) * (1 - np.power(self.a1, 2))
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0) / m
        
        # Update the momentum terms
        self.v_W1 = momentum * self.v_W1 - learning_rate * dW1
        self.v_b1 = momentum * self.v_b1 - learning_rate * db1
        self.v_W2 = momentum * self.v_W2 - learning_rate * dW2
        self.v_b2 = momentum * self.v_b2 - learning_rate * db2
        
        # Update the weights and biases
        self.W1 += self.v_W1
        self.b1 += self.v_b1
        self.W2 += self.v_W2
        self.b2 += self.v_b2
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Generate sample data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize the neural network
input_size = X.shape[1]
hidden_size = 4
output_size = 1
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Training parameters
epochs = 10000
learning_rate = 0.1
momentum = 0.9

# Training loop
for epoch in range(epochs):
    # Forward pass
    y_pred = nn.forward(X)
    
    # Backward pass and weight update
    nn.backward(X, y, learning_rate, momentum)
    
    # Print the loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean((y_pred - y) ** 2)
        print(loss)

#print(y_pred)'''