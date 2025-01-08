import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the neural network with weights and biases.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output neurons.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        """Applies the sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Calculates the derivative of the sigmoid function."""
        return x * (1 - x)

    def forward(self, X):
        """
        Performs a forward pass through the network.

        Args:
            X (np.ndarray): Input data.

        Returns:
            tuple: Output of the hidden layer and the final output.
        """
        # Hidden layer computation
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Output layer computation
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)

        return self.hidden_output, self.output

    def backward(self, X, y, learning_rate):
        """
        Performs backpropagation and updates weights and biases.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): True labels.
            learning_rate (float): Learning rate for weight updates.
        """
        # Compute output error
        output_error = y - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)

        # Compute hidden layer error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        """
        Trains the neural network using the given data.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): True labels.
            epochs (int): Number of training iterations.
            learning_rate (float): Learning rate for weight updates.
        """
        for epoch in range(epochs):
            # Forward pass
            _, output = self.forward(X)

            # Backward pass
            self.backward(X, y, learning_rate)

            # Optional: Print loss for every 100 epochs
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Makes predictions using the trained network.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted output.
        """
        _, output = self.forward(X)
        return output


# Example usage
if __name__ == "__main__":
    # Create dataset (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Initialize neural network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    # Train the network
    nn.train(X, y, epochs=10000, learning_rate=0.1)

    # Test the network
    predictions = nn.predict(X)
    print("Predictions:", predictions)