#Sistemas Inteligentes para Bioinformática

from data import Dataset, summary
import os


DIR = os.path.dirname(os.path.realpath('.'))
filename = os.path.join(DIR, 'datasets/breast-bin.data')
print(filename)


#Labeled dataset

dataset = Dataset.from_data(filename, labeled=True)
print(dataset.X[:5, :])
print(dataset.Y[:5])

print("Has label:", dataset.hasLabel())
print("Number of features:", dataset.getNumFeatures())
print("Number of classes:", dataset.getNumClasses())
print(summary(dataset))

print(dataset.toDataframe())

#Standard Scaler

from util import StandardScaler
sc = StandardScaler()
ds2 = sc.fit_transform(dataset)
print(summary(ds2))

#Feature Selection


from data.feature_selection import f_regress, SelectKBest, VarianceThreshold

#Variance Threshold

vt = VarianceThreshold(8)
ds2 = vt.fit_transform(dataset)
print(summary(ds2))

#SelectKBest

# SelectKBest for classification
skb = SelectKBest(5)
ds3 = skb.fit_transform(dataset)
print(summary(ds3))

#Clustering

from src.si.unsupervised.Clustering import Kmeans
import pandas as pd
import matplotlib.pyplot as plt


# o dataset iris nao estava inicialmente no github
filename = os.path.join(DIR, 'datasets/iris.data')
df = pd.read_csv(filename)
iris = Dataset.from_dataframe(df,ylabel="class")

# indice das features para o plot
c1 = 0
c2 = 1
# plot
plt.scatter(iris.X[:,c1], iris.X[:,c2])
plt.xlabel(iris._xnames[c1])
plt.ylabel(iris._xnames[c2])
plt.show()


kmeans = Kmeans(3)
cent, clust = kmeans.fit_transform(iris)


plt.scatter(iris.X[:,c1], iris.X[:,c2],c=clust)
plt.scatter(cent[:,c1],cent[:,c2], s = 100, c = 'black',marker='x')
plt.xlabel(iris._xnames[c1])
plt.ylabel(iris._xnames[c2])
plt.show()
# podem obter clusterings diferentes já que estes dependem da escolha dos centroids iniciais~

#PCA

from src.si.unsupervised.Clustering import PCA
pca = PCA(2, using='svd')

reduced = pca.fit_transform(iris)[0]
print(pca.explained_variances())

iris_pca = Dataset(reduced,iris.Y,xnames=['pc1','pc2'],yname='class')
iris_pca.toDataframe()


plt.scatter(iris_pca.X[:,0], iris_pca.X[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
