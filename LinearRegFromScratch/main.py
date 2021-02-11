"""
    For now, this is still in progress
    Only works with one coefficient
"""

import numpy as np
import matplotlib.pyplot as plt

X = [1, 2, 4, 3, 5]
y = [1, 3, 3, 2, 5]

lim = 0.9
lr = 0.02  # learning rate

class LinearRegression():
    def __init__(self):
        self.bias = 1

    def fit(self, X, y):
        acc = 0
        if type(X[0]) == int:
            self.coefs = 1
            while acc < lim:
                predictions = self.predict(X)
                y_diff = [a - b for (a, b) in zip(predictions, y)]
                squares = sum([a**2 for a in y_diff])
                s = squares * lr / 2
                self.bias -= s
                self.coefs -= s * np.mean(X)
                mean = sum((y - np.mean(y))**2)
                curr_acc = 1 - squares/mean
                if acc < curr_acc:
                    break
                acc = curr_acc
        else:
            curr_acc = 0
            m = len(X)
            self.coefs = [1 for i in range(len(X))]
            while curr_acc < lim:
                predictions = self.predict(X)
                for (xi, y_pred) in zip(X, predictions):
                    if type(xi) == int:
                        l = 1
                    else:
                        l = len(xi)
                    new_coefs = np.ones(len(X))
                    s = [(y_pred - y_exp) ** 2 for y_exp in y]
                    #s = s * lr / (2 * m)
                    #self.bias -= s
                    for i in range(l):
                        new_coefs[i] -= s
                self.coefs = new_coefs

    def predict(self, X):
        y_pred = []
        for xi in X:
            x_curr = np.dot(self.coefs, xi) + self.bias
            y_pred.append(x_curr)
        return y_pred


reg = LinearRegression()
reg.fit(X, y)
l = np.linspace(0, 7, 100)
predicted = reg.predict(l)
plt.scatter(X, y)
plt.plot(l, predicted)
plt.show()
