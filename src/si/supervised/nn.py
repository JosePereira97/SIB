from abc import abstractclassmethod, ABC
from typing import MutableSequence
from .model import Model
from scipy import signal
import numpy as np
from util.metrics import mse, mse_prime 



class Layer (ABC):

    def __init__(self):
        self.input = None
        self.output = None

    @abstractclassmethod
    def forward(self,input):
        raise NotImplementedError

    @abstractclassmethod
    def backward(self, output_error, learning_rate):
        raise NotImplementedError

class Dense(Layer):

    def __init__(self, input_size, output_size):
    """Fully Connected layer"""
    self.weights = np.random.rand(input_size, output_size) - 0.5
    self.bias = np.zeros((1, output_size))

    def setWeights(self, weights, bias):
        """Sets the weights for the NN.
        :param weights: An numpy.array of weights
        :param bias: the numpy array of bias weights"""

        if(weights.shape != self.weights.shape):
            raise ValueError(F"Shapes mismatch{weights.shape} and")
        
        if(bias.shape != self.bias.shape):
            raise ValueError(F"Shapes mismatch {bias.shape} and")
        self.weights = weights
        self.bias = bias

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_Error, learning_data):
        raise NotImplementedError
    



    
class Activation(Layer):

    def __init__(self, activation):
        self.activation = activation

    def forward (self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    
    def backward (self, output_error, learning_rate):
        #learning_rate is not used because there is no "learnable" parameters.
        #Only passed the error fo the previous layer
        raise np.multiply(self.activation.prime(self.input), output_error)





class NN(Model):

    def __init__(self, epochs = 1000, lr = 0.001, verbose = True):
        """Neural network Model."""
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose

        self.layers = []
        self.loss = mse
        self.loss_prime = mse_prime

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, dataset):
        raise NotImplementedError
    
    def predict(self, input_data):
        assert self.is_fitted, "Model must be fit before prefict"
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def cost(self, X = None, y = None):
        assert self.is_fitted, "Model must be fit before predict"
        x = X if X is not None else self.dataset.X
        y = y if y is not None else self.datase.Y
        output = self.predict(X)
        return self.loss(y, output)

