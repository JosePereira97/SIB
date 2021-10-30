import numpy as np
import pandas as pd

from si.util.util import label_gen

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
        self._xnames = xnames if xnames else label_gen(X.shape[1])
        self._yname = yname if yname else 'Y'

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
        data = np.genfromtxt(filename, delimiter=sep)
        if labeled:
            X = data[:, 0:-1]
            Y = data[:, -1]
        else:
            X = data
            Y = None
        return cls(X, Y)

    @classmethod
    def from_dataframe(cls, df, ylabel=None):
        """Creates a DataSet from a pandas dataframe.

        :param df: [description]
        :type df: [type]
        :param ylabel: [description], defaults to None
        :type ylabel: [type], optional
        :return: [description]
        :rtype: [type]
        """
        if ylabel is not None and ylabel in df.colums:
            X = df.loc[:, df.colums != ylabel].to_numpy()
            Y = df.loc[:, df.colums == ylabel].to_numpy()
            # ou df.loc[:,ylabel].to_numpy()
            Xnames = df.colums.tolist().remove(ylabel)
            Yname = ylabel
        else:
            X= df.to_numpy()
            Y = None
            Xnames= df.colums.tolist()
            Yname = None
        return cls(X,Y,Xnames,Yname)

    def __len__(self):
        """Returns the number of data points."""
        return self.X.shape[0]

    def hasLabel(self):
        """Returns True if the dataset constains labels (a dependent variable)"""
        return self.Y is not None

    def getNumFeatures(self):
        """Returns the number of features (numero de colunas de X)"""
        return self.X.shape[1]
    # ou
    #    return len(self.X), neste caso era necessario previnir o colunas X sem nomes

    def getNumClasses(self):
        """Returns the number of label classes or 0 if the dataset has no dependent variable."""
        unique = np.unique(self.Y, return_counts=False)
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

    def toDataframe(self):
        """ Converts the dataset into a pandas DataFrame"""
        import pandas as pd
        if self.Y is None:
            dataset = pd.DataFrame(self.X.copy(), columns=self._xnames[:])
        else:
            dataset = pd.DataFrame(np.hstack((self.X, self.Y.reshape(len(self.Y), 1))), columns=np.hstack((self._xnames, self._yname)))
        return dataset

    def getXy(self):
        return self.X, self.Y

def summary(dataset, format='df'):
    """ Returns the statistics of a dataset(mean, std, max, min)

    :param dataset: A Dataset object
    :type dataset: si.data.Dataset
    :param format: Output format ('df':DataFrame, 'dict':dictionary ), defaults to 'df'
    :type format: str, optional
    """
    if dataset.hasLabel():
        data = np.hstack((dataset.X,dataset.Y.reshape(len(dataset.Y))))
        names= [dataset._xnames,dataset._yname]
    else:
        data = np.hstack((dataset.X, dataset.Y.reshape(len(dataset.Y))))
        names = [dataset._xnames]
    mean = np.mean(data, axis=0)
    var = np.var(data, axis=0)
    maxi = np.max(data, axis=0)
    mini = np.min(data, axis=0)
    stats = {}
    for i in range(data.shape[1]):
        stat = {'mean' : mean[i]
                ,'var' : var[i]
                ,'max' : maxi[i]
                ,'min' : mini[i]}
        stats[names[i]] = stat
    if format == 'df':
        import pandas as pd
        df= pd.DataFrame(stats)
        return df
    else:
        return stats