import itertools
import numpy as np

# Y is reserved to idenfify dependent variables
ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXZ'

__all__ = ['label_gen', 'summary']


def label_gen(n):
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


def summary(dataset, format='df'):
    """ Returns the statistics of a dataset(mean, std, max, min)

    :param dataset: A Dataset object
    :type dataset: si.data.Dataset
    :param format: Output format ('df':DataFrame, 'dict':dictionary ), defaults to 'df'
    :type format: str, optional
    """
    if dataset.haslabel():
        fuilds = np.hstack((dataset.X, dataset.Y.reshape(len(dataset.y))))
        columns = dataset
    else:
        fuilds = dataset.X
        columns = dataset._xnames[:]
    _means = np.mean(fuilds, axis = 0)
    _vars = np.var(fuilds, axis = 0)
    _maxs = np.max(fuilds, axis = 0)
    _mins = np.min(fuilds, axis = 0)
    statistics = {}
    for i in range(fuilds.shape[1]):
        ##falta acabar
        pass
    

def l2_distance(x,y):
    dist = ((x - y)**2).sum(axis=1)
    return dist

