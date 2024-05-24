import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

class OurMLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        

        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000):
        y = y.reshape(-1, 1)  
        input_size = X.shape[1]  
        
        
        self.weights1 = np.random.randn(input_size, self.hidden_size)
        

        for epoch in range(epochs):
            
            layer1 = np.dot(X, self.weights1)
            activation1 = self.sigmoid(layer1)

            layer2 = np.dot(activation1, self.weights2)
            activation2 = self.sigmoid(layer2)
            
            error_output = 0.5 * (y - activation2) ** 2
            
            delta_output = activation2 - y
            delta_output *= self.sigmoid_derivative(activation2)
            delta_weights2 = np.dot(activation1.T, delta_output)

            delta_hidden = np.dot(delta_output, self.weights2.T)
            delta_hidden *= self.sigmoid_derivative(activation1)
            delta_weights1 = np.dot(X.T, delta_hidden)

            self.weights1 -= self.learning_rate * delta_weights1
            self.weights2 -= self.learning_rate * delta_weights2
            
    def predict(self, X):
        layer1 = np.dot(X, self.weights1)
        activation1 = self.sigmoid(layer1)

        layer2 = np.dot(activation1, self.weights2)
        activation2 = self.sigmoid(layer2)

        return (activation2 > 0.5).astype(int)