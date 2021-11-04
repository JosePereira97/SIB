import numpy as np
import pandas as pd
from util.util_mal import label_gen


__all__ = ['Dataset']


class Dataset:
    def __init__(self, X=None, Y=None,
                 xnames: list = None,
                 yname: str = None):
        """ Tabular Dataset"""
        if X is None:
            raise Exception("Trying to instanciate a DataSet without any data")
        self.X = X
        self.Y = Y
        if xnames:#vamos criar o nome das colunas consoante o exel
            self._xnames = xnames
        else:
            self._xnames = label_gen(X.shape[1])
        self._yname = yname if yname else 'Y' #se n tiver nome a variavel dependente dá se o nome de Y

    @classmethod
    def from_data(cls, filename, sep=",", labeled=True):
        """Creates a DataSet from a data file.
        :cls: classe a qual aplicar o metodo
        :param filename: The filename
        :type filename: str
        :param sep: attributes separator, defaults to ","
        :type sep: str, optional
        :return: A DataSet object
        :rtype: DataSet
        """
        data = np.genfromtxt(filename, delimiter=sep) #carrega dados de um TXT
        if labeled: #se tiver nomes na tabela
            X = data[:, 0:-1] #vai bucar as colunas do Array np exepto a ultima
            Y = data[:, -1] #Vai buscar o valor das variaveis dependentes
        else:
            X = data #se não for label os dados vão estar todos em dentro de um array
            Y = None
        return cls(X, Y) #retornar uma classe com os dados do np array

    @classmethod
    def from_dataframe(cls, df, ylabel=None): #cria um dataset a partir de pandar dataframe
        """Creates a DataSet from a pandas dataframe.

        :param df: [description]
        :type df: [type]
        :param ylabel: [description], defaults to None
        :type ylabel: [type], optional
        :return: [description]
        :rtype: [type]
        """
        if ylabel is not None and ylabel in df.columns:
            X = df.loc[:, df.columns != ylabel].to_numpy() #vai bucar os valores todos que n sejam
            Y = df.loc[:, df.columns == ylabel].to_numpy() #vai buscar os valores das variaveis dependentes
            # ou df.loc[:,ylabel].to_numpy()
            Xnames = df.columns.tolist().remove(ylabel) #nome das colunas todos os que n tiverem em y_label
            Yname = ylabel#nome das linhas todos os que tiverem em y_label
        else:
            X= df.to_numpy()
            Y = None
            Xnames= df.columns.tolist()
            Yname = None
        return cls(X,Y,Xnames,Yname) #devolve a classe com os dados X,Y e os nomes de colunas e linhas

    def __len__(self):
        """Returns the number of data points."""
        return self.X.shape[0] #devolve o numero de linhas do np array

    def hasLabel(self):
        """Returns True if the dataset constains labels (a dependent variable)"""
        return self.Y is not None #numero de lables dependentes

    def getNumFeatures(self):
        """Returns the number of features (numero de colunas de X)"""
        return self.X.shape[1] #numero de colunas dos dados

    def getNumClasses(self):
        """Returns the number of label classes or 0 if the dataset has no dependent variable."""
        unique = np.unique(self.Y, return_counts=False) #numero de variaveis dependentes
        return len(unique) if self.hasLabel() else 0

    def writeDataset(self, filename, sep=","):
        """Saves the dataset to a file

        :param filename: The output file path
        :type filename: str
        :param sep: The fields separator, defaults to ","
        :type sep: str, optional
        """
        fullds = np.hstack((self.X, self.Y.reshape(len(self.Y), 1)))
        np.savetxt(filename, fullds, delimiter=sep)

    def toDataframe(self): #convert um dataset num dataframe
        """ Converts the dataset into a pandas DataFrame"""

        if self.Y is None:
            dataset = pd.DataFrame(self.X.copy(), columns=self._xnames[:])
        else:
            dataset = pd.DataFrame(np.hstack((self.X, self.Y.reshape(len(self.Y), 1))), columns=np.hstack((self._xnames, self._yname)))
        return dataset

    def getXy(self): #buscar os dados
        return self.X, self.Y

def summary(dataset, format='df'):
    """ Returns the statistics of a dataset(mean, std, max, min)

    :param dataset: A Dataset object
    :type dataset: si.data.Dataset
    :param format: Output format ('df':DataFrame, 'dict':dictionary ), defaults to 'df'
    :type format: str, optional
    """
    if dataset.hasLabel():
        data = np.hstack((dataset.X,dataset.Y.reshape(-1,1))) #se tiver label em variaveis dependentes
        names= []
        for i in dataset._xnames:
            names.append(i)
        names.append(dataset._yname)
    else:
        data = np.hstack((dataset.X, dataset.Y.reshape(len(dataset.Y))))
        names = [dataset._xnames]
    mean = np.mean(data, axis=0) #calculo da média
    var = np.var(data, axis=0) #calculo da variância
    maxi = np.max(data, axis=0) #calculo do máximo
    mini = np.min(data, axis=0) #calculo do minimo
    stats = {}
    for i in range(data.shape[1]): #guardar tudo numa lista
        stat = {'mean' : mean[i]
                ,'var' : var[i]
                ,'max' : maxi[i]
                ,'min' : mini[i]}
        stats[names[i]] = stat
    if format == 'df': #tramsformar em um dataset
        df= pd.DataFrame(stats)
        return df
    else:
        return stats