import numpy as np
from abc import ABC, abstractmethod
from .model import Model
from ..util.activation import *
from ..util.metrics import mse, mse_prime
from ..util.im2col import pad2D, im2col, col2im


class Layer(ABC):

    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input):
        raise NotImplementedError

    @abstractmethod
    def backward(self, erro, learning_rate):
        raise NotImplementedError


class Dense(Layer):

    def __init__(self, input_size, output_size):
        super(Dense, self).__init__()
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.zeros((1, output_size))

    def setWeights(self, weights, bias):
        if (weights.shape != self.weights.shape):
            raise ValueError(f"Shapes mismatch {weights.shape} and {self.weights.shape}")
        if (bias.shape != self.bias.shape):
            raise ValueError(f"Shapes mismatch {bias.shape} and {self.bias.shape}")
        self.weights = weights
        self.bias = bias

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        weights_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0)
        input_error = np.dot(output_error, self.weights.T)

        # Update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return input_error


class Activation(Layer):

    def __init__(self, func):
        super(Activation, self).__init__()
        self.activation = func

    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        return np.multiply(self.activation.prime(self.input), output_error)


class NN(Model):

    def __init__(self, epochs=1000, lr=0.01, verbose=True):
        super(NN, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose

        self.layers = []
        self.loss = mse
        self.loss_prime = mse_prime

    def useLoss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, dataset):
        self.dataset = dataset
        X, y = dataset.getXy()
        self.history = dict()
        for epoch in range(self.epochs):
            output = X
            # forward propagation
            for layer in self.layers:
                output = layer.forward(output)

            # backward propagation
            error = self.loss_prime(y, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, self.lr)

            # calculate average error on all samples
            err = self.loss(y, output)
            self.history[epoch] = err
            if self.verbose:
                print(f"epoch {epoch+1}/{self.epochs}, error= {err}")
            else:
                print(f"epoch {epoch + 1}/{self.epochs}, error= {err}", end='\r')

    def fit_batch(self, dataset, batchsize=256):
        X, y = dataset.getXy()
        if batchsize > X.shape[0]:
            raise Exception('Number of batchs superior to length of dataset')
        n_batches = int(np.ceil(X.shape[0] / batchsize))
        self.dataset = dataset
        self.history = dict()
        for epoch in range(self.epochs):
            self.history_batch = np.zeros((1, batchsize))
            for batch in range(n_batches):
                output = X[batch * batchsize:(batch + 1) * batchsize, ]

                # forward propagation
                for layer in self.layers:
                    output = layer.forward(output)

                # backward propagation
                error = self.loss_prime(y[batch * batchsize:(batch + 1) * batchsize, ], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, self.lr)

                # calcule average error
                err = self.loss(y[batch * batchsize:(batch + 1) * batchsize, ], output)
                self.history_batch[0, batch] = err
            self.history[epoch] = np.average(self.history_batch)
            if self.verbose:
                print(f'epoch {epoch + 1}/{self.epochs}, error = {self.history[epoch]}')
            else:
                print(f"epoch {epoch + 1}/{self.epochs}, error = {self.history[epoch]}", end='\r')
        self.is_fitted = True #não estava a funcionar sem quebrar o ciclo for em qualquer ponto

    def predict(self, x):
        assert self.is_fitted, "Model must be fitted before prediction"
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def cost(self, X=None, Y=None):
        assert self.is_fitted, "Model must be fitted before prediction"
        X = X if X is not None else self.dataset.X
        Y = Y if Y is not None else self.dataset.Y
        output = self.predict(X)
        return self.loss(Y, output)


class Flatten(Layer):

    def forward(self, input):
        self.input_shape = input.shape
        # flattern all but the 1st dimension
        output = input.reshape(input.shape[0], -1)
        return output

    def backward(self, erro, learning_rate):
        return erro.reshape(self.input_shape)


class Conv2D(Layer):

    def __init__(self, input_shape, kernel_shape, layer_depth, stride=1, padding=0):
        self.input_shape = input_shape
        self.in_ch = input_shape[2]
        self.out_ch = layer_depth
        self.stride = stride
        self.padding = padding
        # weights
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], self.in_ch, self.out_ch) - 0.5
        # bias
        self.bias = np.zeros((self.out_ch, 1))

    def forward(self, input):
        s = self.stride
        self.X_shape = input.shape
        _, p = pad2D(input, self.padding, self.weights.shape[:2], s)

        pr1, pr2, pc1, pc2 = p
        fr, fc, in_ch, out_ch = self.weights.shape
        n_ex, in_rows, in_cols, in_ch = input.shape

        # compute the dimensions of the convolution output
        out_rows = int((in_rows + pr1 + pr2 - fr) / s + 1)
        out_cols = int((in_cols + pc1 + pc2 - fc) / s + 1)

        # convert X and W into the appropriate 2D matrices and take their product
        self.X_col, _ = im2col(input, self.weights.shape, p, s)
        W_col = self.weights.transpose(3, 2, 0, 1).reshape(out_ch, -1)

        output_data = (W_col @ self.X_col + self.bias).reshape(out_ch, out_rows, out_cols, n_ex).transpose(3, 1, 2, 0)

        return output_data

    def backward(self, erro, learning_rate):

        fr, fc, in_ch, out_ch = self.weights.shape
        p = self.padding
        db = np.sum(erro, axis=(0, 1, 2))
        db = db.reshape(out_ch,)

        dout_reshaped = erro.transpose(1, 2, 3, 0).reshape(out_ch, -1)
        dW = dout_reshaped @ self.X_col.T
        dW = dW.reshape(self.weights.shape)

        W_reshape = self.weights.reshape(out_ch, -1)
        dX_col = W_reshape.T @ dout_reshaped
        input_error = col2im(dX_col, self.X_shape, self.weights.shape, (p, p, p, p), self.stride)

        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

        return input_error

class Pooling2D(Layer):

    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def pool(X_col):  # self?
        raise NotImplementedError

    def dpool(dX_col, dout_col, pool_cache):  # self?
        raise NotImplementedError

    def forward(self, input):
        self.X_shape = input.shape
        n, h, w, d = input.shape
        h_out = (h - self.size) / self.stride + 1
        w_out = (w - self.size) / self.stride + 1

        if not w_out.is_integer() or not h_out.is_integer():
            raise Exception('Invalid output dimension')

        h_out, w_out = int(h_out), int(w_out)

        X_reshaped = input.reshape(n * d, h, w, 1)

        self.X_col, _ = im2col(X_reshaped, (self.size, self.size, d, d), pad=0, stride=self.stride)


        out, self.max_idx = self.pool(self.X_col)

        out = out.reshape(d, h_out, w_out, n)
        out = out.transpose(3, 1, 2, 0)

        return out

    def backward(self, erro, learning_rate):
        n, w, h, d = self.X_shape

        dX_col = np.zeros_like(self.X_col)
        dout_col = erro.transpose(1, 2, 3, 0).ravel()

        dX = self.dpool(dX_col, dout_col, self.max_idx)

        dX = col2im(dX, (n * d, h, w, 1), (self.size, self.size, d, d), pad=0, stride=self.stride)
        dX = dX.reshape(self.X_shape)

        return dX

class MaxPooling2D(Pooling2D):

    def pool(self,X_col):
        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.size)]
        return out, max_idx

    def dpool(self,dX_col, dout_col, pool_cache):
        dX_col[pool_cache, range(dout_col.size)] = dout_col
        return dX_col

    def __str__(self):
        return "MaxPooling2D"

