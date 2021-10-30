import numpy as np
from src.si.util.util import euclidean,manhattan
import warnings


class Kmeans:
    def __init__(self, k :int, itera = 1000, dist = 'euclidean'):
        self.k = k
        self.itera = itera
        if dist is 'euclidean':
            self.dist = euclidean
        elif dist is 'manhattan':
            self.dist = manhattan
        else:
            raise Exception('Distance metric not available \n Score functions: euclidean, manhattan')

    def fit(self,dataset):
        X = dataset.X
        self.min = np.min(X,axis = 0)
        self.max = np.max(X, axis = 0)

    def initcentroids(self,dataset):
        X = dataset.X
        self.centroids = np.array([np.random.uniform(low=self.min[i], high=self.max[i], size=(self.k,)) for i in range(X.shape[1])]).T

    def closest_centroid(self,x):
        dist = euclidean(x,self.centroids)
        closest_centroid_ind = np.argmin(dist,axis = 0)
        return closest_centroid_ind

    def fit_transform(self,dataset):
        self.fit(dataset)
        centroides, idxs = self.transform(dataset)
        return centroides, idxs

    def transform(self,dataset):
        self.initcentroids(dataset)
        print(self.centroids)
        X = dataset.X
        change = True
        count = 0
        old_ind = np.zeros(X.shape[0])
        while change or count < self.itera:
            idxs = np.apply_along_axis(self.closest_centroid,axis = 0, arr = X.T)
            cent = []
            for i in range(self.k):
                cent.append(np.mean(X[idxs == i], axis = 0))
            self.centroids = np.array(cent)
            change = np.all(old_ind == idxs)
            old_ind = idxs
            count += 1
        return self.centroids,old_ind


class PCA:
    def __init__(self, n_components=2, using="svd"):
        self.n_components = n_components
        self.using = using

    def fit(self, dataset):
        pass

    def transform(self, dataset):
        from sklearn.preprocessing import StandardScaler
        x = dataset.X
        X_scaled = StandardScaler().fit_transform(x)  #Normalização por standart scaler
        features_scaled = X_scaled.x.T
        if self.using == "svd":
            self.vectors, self.values, rv = np.linalg.svd(features_scaled)
        else:
            cov_matrix = np.cov(features_scaled)
            self.values, self.vectors = np.linalg.eig(cov_matrix)
        self.sorted_comp = np.argsort(self.values)[::-1]  #gera uma lista com os idexs das colunas ordenadas por importancia de componte
        self.s_value = self.values[self.sorted_comp]   #colunas dos valores e vetore sao reordenadas pelos idexes das colunas
        self.s_vector = self.vectors[:, self.sorted_comp]
        if self.n_components not in range(0,x.shape[1]+1):
            warnings.warn('Number of components is not between 0 and '+str(x.shape[1]))
            self.n_components = x.shape[1]
            warnings.warn('Number of components defined as ' + str(x.shape[1]))
        self.vetor_subset = self.s_vector[:, 0:self.n_components] #gera um conjunto apartir dos vetores e values ordenados
        X_reduced = np.dot(self.vetor_subset.transpose(), features_scaled).transpose()
        return X_reduced

    def explained_variances(self):
        self.values_subset = self.s_value[:, 0:self.n_components]
        return np.sum(self.values_subset), self.values_subset

    def fit_transform(self,dataset):
        x_reduced = self.transform(dataset)
        e_var, vari = self.explained_variances()
        return x_reduced, e_var, vari

