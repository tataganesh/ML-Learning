import numpy as np
# from sklearn.datasets import load_boston
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from celluloid import Camera
from sklearn.utils import shuffle as sklearn_shuffle
from IPython.display import HTML
np.random.seed(123)


def least_squares(X, y):
    N = X.shape[0]
    X = np.squeeze(X)
    slope = (N*np.sum(X*y) - np.sum(X) * np.sum(y))/(N*np.sum(np.square(X)) - np.square(np.sum(X)))
    intercept = (np.sum(y) - slope * np.sum(X)) / N
    return slope, intercept


class simple_sgd:
    def __init__(self):
        # self.all_slopes = [[self.intercept]]
        # self.all_intercepts = [[self.slope]]
        pass

    def fit(self, X, y, batch_size = 2, max_iterations = 3, shuffle = False, lr = 0.1):
        num_features = X.shape[1]
        self.w = np.random.normal(size=num_features) * 20
        self.b = np.random.normal()
        if shuffle:
            X, y = sklearn_shuffle(X, y, random_state=0)
        num_samples = X.shape[0]
        num_iterations = 0
        while (num_iterations < max_iterations):
            if num_iterations > 20:
                lr = 0.001
            for i in range(0, num_samples, batch_size):
                batch_x = X[i: i + batch_size]
                batch_y = y[i: i + batch_size]
                y_pred = np.dot(batch_x, self.w) + self.b
                self.w = self.w + 2 * lr * np.mean((batch_y - y_pred)[:, np.newaxis] * batch_x, axis=0)
                self.b += 2 * lr * np.mean(batch_y - y_pred)
                # self.all_slopes.append(self.slope)
                # self.all_intercepts.append(self.intercept)
            # m, b = least_squares(X, y)
            error = np.mean(np.square(y - (np.dot(X,  self.w) + self.b)))
            # lsError = np.mean(np.square(y - np.squeeze(m*X + b)))
            print(f"Epoch  - {num_iterations} , Error - {error}")
            num_iterations += 1
            if error < 0.01:
                break

    def predict(self):
        pass

    def visualize(self, X, y):
        pass

bias = 10
X, y, coefs = make_regression(n_samples = 500, n_features = 10, noise=10, coef=True, bias = bias)

lin_reg = simple_sgd()
lin_reg.fit(X, y, batch_size = 5, max_iterations=30, lr = 0.1)
print(f"Computed weights - {lin_reg.w}, Bias - {lin_reg.b}")
print(f"Actual weights - {coefs}, Bias - {bias}")
