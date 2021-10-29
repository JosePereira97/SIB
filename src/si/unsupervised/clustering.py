import numpy as np

# realizar calsse de PCA

class KMeans:

    def __init__(self, k: int, max_interations = 100):
        self.k = k
        self.max_interations = max_interations

    def fit(sels,dataset):
        """randomly select k centroids
        """
        x = dataset.X
        self._min = np.min(x,axis=0)
        self._max = np.max(x,axis=0)

    def init_centroids(self, dataset):
        x = dataset.X
        self.centroids = np.array(
            [np.random.uniform(
                low = self._min[i], high = self._max[i], size = (self.k,)
            ) for i in range (x.shape[1])]).T
        
    def get_closest_centroid(self,x):
        dist = self.distance(x, self.centroids)
        closest_centroid_index = np.argmin(dist, axis=0)
        return closest_centroid_index
    
    def transform(self, dataset):
        self.init_centroids(dataset)
        print(self.centroids)
        X = dataset.X
        changed = True
        count = 0
        while changed or count < self.max_interactions:
            idxs = np.apply_along_axis(self.get_closest_centroid, axis = 0, arr=X.T)
            cent = []
            for i in range(self.k):
                cent.append(np.mean(X[idxs == i], axis = 0))
            self.centroids = np.array(cent)
            changed = np.all(old_idxs == idxs)
            old_idxs = idxs
            count +=1
        return self.centroids, idxs



