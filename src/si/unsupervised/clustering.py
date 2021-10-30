import numpy as np
import scipy.stats as stats

class PCA:
    def __init__(self, n_components=2, using="svd"):
        self.n_components = n_components
        self.using = using

    def fit(self, dataset):
        pass
    
    def transform(self, dataset):
        x = dataset.X
        n, p = x.shape

        scale = StandardScaler().fit_transform(dataset)
        x_center = scale.x

        if self.using == "svd":
            self.s_vecs, self.s_vals, rv = np.linalg.svd(x_center.T)
        else:
            cov_matrix = np.cov(x_center.T)
            self.s_vals, self.s_vecs = np.linalg.eig(cov_matrix)
        self.sorted_idx = np.argsort(self.s_vals)[::-1]
        self.sorted_eigenvalue = self.s_vals[self.sorted_idx]
        self.sorted_eigenvectors = self.s_vecs[:, self.sorted_idx]
        self.eigenvector_subset = self.sorted_eigenvectors[:, 0:self.n_components]
        x_red = np.dot(self.eigenvector_subset.transpose(), x_center.transpose()).transpose()
        return x_red


class Kmeans:
    def __init__(self, k :int, itera = 1000):
        self.k = k
        self.itera = itera

    def fit(self,dataset):
        X = dataset.X
        self.min = np.min(X,axis = 0)
        self.max = np.max(X, axis = 0)

    def initcentroids(self,dataset):
        X = dataset.X
        self.centroids = np.array([np.random.uniform(low=self.min[i], high=self.max[i], size=(self.k,)) for i in range(X.shape[1])]).T
        #tentar modo shuffle

    def closest_centroid(self,x):
        dist = distance(x,self.centroids)
        closest_centroid_ind = np.argmin(dist,axis = 0)
        return closest_centroid_ind

    def fit_transform(self,dataset):
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
            #cent= [np.mean(X[idxs == i], axis = 0) for i in range(self.k)] lista de compreensão
            self.centroids = np.array(cent)
            change = np.all(old_ind == idxs)
            old_ind = idxs
            count += 1
        return self.centroids,idxs



