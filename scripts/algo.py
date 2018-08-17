from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

def kmeans(dataset):
    results = KMeans(n_clusters=2, random_state=0).fit(dataset)
    return results

def knn(dataset):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(dataset)