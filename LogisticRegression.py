import numpy as np

class OurLogisticRegression():


    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        nsamples, nfeatures = X.shape
        self.weights = np.zeros(nfeatures)
        self.bias = 0


        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)

            dw = (1 / nsamples) * np.dot(X.T, (predictions - y))
            db = (1 / nsamples) * np.sum(predictions - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)

        class_pred = [1 if y >= 0.5 else 0 for y in y_pred]
        return class_pred