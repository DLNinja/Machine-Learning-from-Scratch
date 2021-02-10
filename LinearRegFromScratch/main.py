"""
    For now, this is still in progress
"""

import numpy as np
import matplotlib.pyplot as plt

X = [1, 2, 4, 3, 5]
y = [1, 3, 3, 2, 5]

lim = 0.9
lr = 0.1  # learning rate

class LinearRegression():
    def __init__(self):
        self.bias = 1
        self.coefs = []

    def calculate(self, x, y):
        curr_acc = 0
        self.coefs = np.ones(len(x))
        m = len(x)
        while curr_acc < lim:
            predictions = self.predict(x)
            for (xi, y_pred) in zip(x, predictions):
                new_coefs = np.ones(len(x))
                s = sum([(y_pred - y_exp)**2 for y_exp in y])
                s = s * lr / (2*m)
                self.bias -= s
                for i in range(len(xi)):
                    self.coefs[i] -= s * xi[i]

            curr_acc = 0

    def fit(self, X, y):
        pass

    def predict(self, X):
        y_pred = []
        for xi in X:
            x_curr = np.dot(self.coefs, xi) + self.bias
            y_pred.append(x_curr)
        return y_pred


plt.scatter(X, y)
plt.show()
