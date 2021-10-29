import numpy as np
from scipy import stats
from copy import copy
import warnings

class VarianceThreshhold:
    def __init__(self, threshold = 0):
        if threshold < 0:
            warnings.warn('The threshold must be non-negative value')
        self.threshold = threshold

    def fit(self, dataset):
        x = dataset.x
        self._var = np.var(X, axis=0)

    def transform(self, dataset, inline = False):
        X = dataset.X
        cond = self._var > self.threshold
        idxs = []
        for i in range(len(cond)):
            if cond[i]:
                idxs.append(i)
        X_trans = X[:, idxs]
        xnames = []
        for m in idxs:
            xnames.append(dataset.X[m])
        if inline:
            dataset.X = X_trans
            dataset.xnames = xnames
            return dataset
        else:
            from.dataset import Dataset
            return Dataset(copy(X_trans)) #falta coisa
