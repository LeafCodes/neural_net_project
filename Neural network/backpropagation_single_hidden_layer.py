import numpy as np

# Define sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define neural network architecture
input_size = 2
hidden_size = 3
output_size = 1

# Initialize weights and biases with small random values
hidden_weights = np.random.randn(input_size, hidden_size)
hidden_biases = np.zeros((1, hidden_size))
output_weights = np.random.randn(hidden_size, output_size)
output_biases = np.zeros((1, output_size))

# Define hyperparameters
learning_rate = 0.1
num_epochs = 10000

# Define training data and labels
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Train the network using backpropagation
for epoch in range(num_epochs):
    # Forward pass
    hidden_activations = sigmoid(np.dot(X, hidden_weights) + hidden_biases)
    output_activations = sigmoid(np.dot(hidden_activations, output_weights) + output_biases)
    
    # Calculate loss and gradients
    error = y - output_activations
    output_gradients = error * sigmoid_derivative(output_activations)
    hidden_gradients = np.dot(output_gradients, output_weights.T) * sigmoid_derivative(hidden_activations)
    
    # Update weights and biases
    output_weights += learning_rate * np.dot(hidden_activations.T, output_gradients)
    output_biases += learning_rate * np.sum(output_gradients, axis=0)
    hidden_weights += learning_rate * np.dot(X.T, hidden_gradients)
    hidden_biases += learning_rate * np.sum(hidden_gradients, axis=0)

    # Print the loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {np.mean(np.abs(error))}")

# Test the network
hidden_activations = sigmoid(np.dot(X, hidden_weights) + hidden_biases)
output_activations = sigmoid(np.dot(hidden_activations, output_weights) + output_biases)
print("Predictions:", output_activations)