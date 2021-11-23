import numpy as np
from si.data.dataset import Dataset
from si.supervised.metrics import accuracy_score





class KNN(Model):

    def __init__(self, number_neighbors, classification = True):
        super(KNN).__init__()
        self.number_neighbors = number_neighbors
        self.classification = classification

    def fit(self, dataset):
        self.dataset = dataset
        self.is_fitted = True
        
    def get_neighbors(self, x):
        distances = l2_distances(x, self.dataset.X)
        sorted_index = np.argsort(distances)
        return sorted_index[:self.number_neighbors]

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predict'
        neighbors = self.get_neighbors(x)
        values = self.dataset.y[neighbors].tolist()
        if self.classification:
            prediction = max(set(values), key = values.count)
        else:
            prediction = sum(values)/len(values)
        return prediction

    def cost(self):
        y_pred = np.ma.apply_along_axis(self.predict, axis = 0, arr = self.dataset.X.T)
        return accuracy_score(self.dataset.Y, y_pred)

    
def train_test_split(dataset, split = 0.8):
    n = dataset.X.shape[0]
    m = int(split*n)
    arr = np.arrange(n)
    np.random.shuffle(arr)
    train = Dataset(dataset.X[arr[:m]], dataset.Y[arr[:m]], dataset._xnames, dataset._yname)
    test = Dataset(dataset[arr[m:]], dataset.Y[arr[m:]], dataset._xnames, dataset._yname)
    return train, test