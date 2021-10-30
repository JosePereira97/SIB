import itertools

# Y is reserved to idenfify dependent variables
import numpy as np

ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXZ'

__all__ = ['label_gen', 'summary']


def label_gen(n): #forma uma lista de colunas igual ao exel "A","B" etc...
    """ Generates a list of n distinct labels similar to Excel"""
    def _iter_all_strings():
        size = 1
        while True:
            for s in itertools.product(ALPHA, repeat=size):
                yield "".join(s)
            size += 1

    generator = _iter_all_strings()

    def gen():
        for s in generator:
            return s

    return [gen() for _ in range(n)]

def euclidean(x,y): #vai calcular a distância euclidiana
    '''compute de distance to a n dimensional vector x to a list of m,n and summing
    x.shape = (n,) y.shape = (m,n)'''
    dist = ((x-y)**2).sum(axis = 1)
    return dist

def manhattan(x, y): #vai calcular a distância de manhattan
    dist = np.abs(x - y)
    dist = np.sum(dist)
    return dist
