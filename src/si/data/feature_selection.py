import numpy as np
from copy import copy
import warnings
from src.si.data.dataset import Dataset
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
        self.var = np.var(X, axis=0) #vai calcular a variância dos valores do dataset

    def transform(self, dataset, inline = False): #Tranform filtrar dados
        X = dataset.X
        cond = self.var > self.threshold
        ind = []
        for i in range(len(cond)):
            if cond[i]:
                ind.append(i) #se a variância for maior do que o threshold
        X_trans = X[:,ind] #bucar os valores que expliques a variância
        xnames = []
        for i in ind:
            xnames.append(dataset._xnames[i]) #buscar os nomes das colunas em que o valor de variância é mairo que o threshhold
        if inline:
            dataset.X = X_trans
            dataset._xnames = xnames
            return dataset
        else:
            return Dataset(X_trans, copy(dataset.Y), xnames, copy(dataset._yname))

    def fit_transform(self, dataset, inline = False):
        self.fit(dataset)
        return self.transform(dataset, inline = inline)



class SelectKBest:
    def __init__(self,k,score_fun = 'f_regression'):
        if score_fun == "f_classif":
            self.score_fun = f_classif
        elif score_fun == "f_regression":
            self.score_fun = f_regress
        else:
            raise Exception("Score function not available \n Score functions: f_classif, f_regression")

        if k <= 0:
            raise Exception("K value invalid. K-value must be >0") #numero de top selecionar
        else:
            self.k = k

    def fit(self, dataset):
        self.Fscore, self.pval= self.score_fun(dataset) #vai buscar os valores da regressão de pearson

    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline=inline) #return do tranform inline, sem os Y

    def transform(self, dataset, inline=False): #selecionar os melhores valores com K
        data = copy(dataset.X)
        names = copy(dataset._xnames)
        if self.k > data.shape[1]:
            warnings.warn('K value greater than the number of features available')
            self.k = data.shape[1]
        #seleção de lista
        sel_list = np.argsort(self.Fscore)[-self.k:]
        Xdata = data[:, sel_list] # tranform data
        Xnames = [names[index] for index in sel_list]
        if inline:
            dataset.X = Xdata
            dataset._xnames = Xnames
            return dataset
        else:
            return Dataset(Xdata, copy(dataset.Y), Xnames, copy(dataset._yname))

def f_classif(dataset): #realizar a anova
    'ANOVA'
    X, Y = dataset.getXy()
    args = []
    for k in np.unique(Y):
        args.append(X[Y == k, :])
    from scipy.stats import f_oneway
    F_stat, pvalue = f_oneway(*args )
    return F_stat, pvalue

def f_regress(dataset): #regressão de pearson
    'Regressao Pearson'
    X, y = dataset.getXy()
    cor_coef = np.array([stats.pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
    dof = y.size - 2  # degrees of freedom
    cor_coef_sqrd = cor_coef ** 2
    F = cor_coef_sqrd / (1 - cor_coef_sqrd) * dof
    from scipy.stats import f
    p = f.sf(F, 1, dof)
    return F, p