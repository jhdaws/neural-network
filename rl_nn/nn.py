import numpy as np

class rlNeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr #change in weights in each step of training

        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))

        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims = True))
        return expZ / np.sum(expZ, axis=0, keepdims = True)

    def forward_propagation(self, X):
        Z1 = self.W1 @ X + self.b1
        A1 = self.relu(Z1)

        Z2 = self.W2 @ A1 + self.b2
        A2 = self.softmax(Z2)

        cache = (X, Z1, A1, Z2, A2)
        return A2, cache

    def compute_loss(self, A2, Y):
        m = Y.shape[1]
        log_probs = -np.log(A2[Y.argmax(axis=0), range(m)])
        return np.sum(log_probs) / m # calculates the average loss    

    def backward(self, cache, Y):
        X, Z1, A1, Z2, A2 = cache
        m = X.shape[1]

        dZ2 = A2 - Y 
        dW2 = dZ2 @ A1.T / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * self.relu_derivative(Z1)
        dW1 = dZ1 @ X.T / m
        db1 = np.sum(dZ1, axis=1, keepdims = True) / m

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, Y, epochs=100):
        loss_history = []
        acc_history = []

        for epoch in range(epochs):
            A2, cache = self.forward_propagation(X)
            loss = self.compute_loss(A2, Y)
            self.backward(cache, Y)

            predictions = np.argmax(A2, axis=0)
            targets = np.argmax(Y, axis=0)
            accuracy = np.mean(predictions == targets)

            loss_history.append(loss)
            acc_history.append(accuracy)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%")

        return loss_history, acc_history

