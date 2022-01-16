import numpy as np
from si.supervised.model import Model


def majority(values):
    return max(set(values), key=values.count)


def average(values):
    return sum(values)/len(values)


class Ensemble(Model):
    def __init__(self, models, fvote, score):
        super(Ensemble, self).__init__()
        self.models = models
        self.fvote = fvote
        self.score = score

    def fit(self, dataset):
        self.dataset = dataset
        for model in self.models:
            model.fit(dataset)
        self.is_fitted = True

    def predict(self, x):
        assert self.is_fitted, 'Model not fitted'
        preds = [model.predict(x) for model in self.models]
        vote = self.fvote(preds)
        return vote

    def cost(self, X=None, Y=None):
        X = X if X is not None else self.dataset.X
        Y = Y if Y is not None else self.dataset.Y

        Y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=X.T)
        return self.score(Y, Y_pred)