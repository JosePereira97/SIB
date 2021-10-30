import numpy as np
from copy import copy
import warnings
from ..data import Dataset
import scipy.stats as stats


class VarianceThreshold:
    def __init__(self,threshold=0):
        '''
        the variance threshold is a simple baseline approach to feature selection
        it removes all features which variance doesn't meet some threshold limit
        it removes all zero-variance features, i.e..
        '''
        if threshold < 0:
            warnings.warn('thr e threshold must be a non negative value')
        self.threshold = threshold

    def fit(self,dataset):
        X = dataset.X
        self.var = np.var(X, axis=0)

    def transform(self, dataset, inline = False):
        X = dataset.X
        cond = self.var > self.threshold
        ind = []
        for i in range(len(cond)):
            if cond[i]: ind.append(i)
        X_trans = X[:,ind]
        xnames = [dataset._xnames[i] for i in ind]
        if inline:
            dataset.X = X_trans
            dataset._xnames = xnames
            return dataset
        else:
            from ..data import Dataset
            Dataset(X_trans, copy(dataset.Y), xnames, copy(dataset.yname))

class SelectKBest:
    def __init__(self,k,score_fun):
        if score_fun == "f_classif":
            self.score_fun = f_classif
        elif score_fun == "f_regression":
            self.score_fun = f_regress
        else:
            raise Exception("Score function not available \n Score functions: f_classif, f_regression")

        if k <= 0:
            raise Exception("K value invalid. K-value must be >0")
        else:
            self.k = k

    def fit(self, dataset):
        self.Fscore, self.pval= self.score_fun(dataset)

    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline=inline)

    def transform(self, dataset, inline=False):
        ordata = copy(dataset.X)
        ornames = copy(dataset.xnames)
        if self.k > ordata.shape[1]:
            warnings.warn('K value greater than the number of features available')
            self.k = ordata.shape[1]
        #seleção de lista
        sel_list = np.argsort(self.Fscore)[-self.k:]
        ndata = ordata[:, sel_list]
        nnames = [ornames[index] for index in sel_list]
        if inline:
            dataset.X = ndata
            dataset.xnames = nnames
            return dataset
        else:
            return Dataset(ndata, copy(dataset.Y), nnames, copy(dataset.yname))

def f_classif(dataset):
    'ANOVA'
    X, y = dataset.getXy()
    args = []
    for k in np.unique(y):
        args.append(X[y == k, :])
    from scipy.stats import f_oneway
    F_stat, pvalue = f_oneway(*args )
    return F_stat, pvalue

def f_regress(dataset):
    'Regressao Pearson'
    X, y = dataset.getXy()
    cor_coef = np.array([stats.pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
    dof = y.size - 2  # degrees of freedom
    cor_coef_sqrd = cor_coef ** 2
    F = cor_coef_sqrd / (1 - cor_coef_sqrd) * dof
    from scipy.stats import f
    p = f.sf(F, 1, dof)
    return F, p