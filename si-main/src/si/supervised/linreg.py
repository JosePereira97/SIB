from si.supervised.model import Model
from ..util.metrics import mse
from ..util.util import add_intersect
import numpy as np


class LinearRegression(Model):

    def __init__(self, gd=False, epochs=1000, lr=0.001):
        super(LinearRegression, self).__init__()
        self.gd = gd
        self.theta = None
        self.epochs = epochs
        self.lr = lr

    def fit(self, dataset):
        X, Y = dataset.getXy()
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.X = X
        self.Y = Y

        # closed form of GD
        self.train_gd(X, Y) if self.gd else self.train_closed(X, Y)
        self.is_fitted = True

    def train_closed(self, x, y):
        self.theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

    def train_gd(self, x, y):
        m = x.shape[0]
        n = x.shape[1]

        self.history = {}  # criar um historico dos thetas e custo por epoch
        self.theta = np.zeros(n)

        for epoch in range(self.epochs):
            grad = 1/m * (x.dot(self.theta) - y).dot(x)
            self.theta -= self.lr * grad

            self.history[epoch] = [self.theta[:], self.cost()]

    def predict(self, x):
        assert self.is_fitted, 'Model must be fitted before predicting'
        x = np.array(x)
        if x.ndim > 1:
            res = []
            for i in x:
                _x = np.hstack(([1], i))
                res.append(np.dot(self.theta, _x))
        else:
            _x = np.hstack(([1], x))
            res = np.dot(self.theta, _x)
        return res

    def cost(self, X=None, Y=None, theta=None):
        X = add_intersect(X) if X is not None else self.X
        Y = Y if Y is not None else self.Y
        theta = theta if theta is not None else self.theta

        y_pred = np.dot(X, theta)
        return mse(Y, y_pred) / 2


class LinearRegressionReg(LinearRegression):

    def __init__(self, gd=False, epochs=1000, lr=0.001, lbd=1):
        super(LinearRegressionReg, self).__init__(gd=gd, epochs=epochs, lr=lr)
        self.lbd = lbd

    def train_closed(self, x, y):
        n = x.shape[1]
        identity = np.eye(n)
        identity[0, 0] = 0
        self.theta = np.linalg.inv(x.T.dot(x) + self.lbd * identity).dot(x.T).dot(y)
        self.is_fitted = True

    def train_gd(self, x, y):
        m = x.shape[0]
        n = x.shape[1]

        self.history = {}  # criar um historico dos thetas por epoch
        self.theta = np.zeros(n)
        lbds = np.full(m, self.lbd)
        lbds[0] = 0

        for epoch in range(self.epochs):
            grad = 1 / m * (x.dot(self.theta) - y).dot(x)
            self.theta -= (self.lr/m) * (lbds+grad)

            self.history[epoch] = [self.theta[:], self.cost()]