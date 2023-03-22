import matplotlib.pyplot as plt
import time

# Dimension reduction and clustering libraries
import umap
import sklearn.cluster as cluster
from sklearn.metrics import normalized_mutual_info_score
from sklearn.utils.linear_assignment_ import linear_assignment

import h5py
import numpy as np

with h5py.File("usps.h5", 'r') as hf:
    train = hf.get('train')
    X_tr = train.get('data')[:]
    y_tr = train.get('target')[:]
    test = hf.get('test')
    X_te = test.get('data')[:]
    y_te = test.get('target')[:]
    d = np.concatenate((X_tr, X_te), axis=0)
    y = np.concatenate((y_tr, y_te), axis=0)

def best_cluster_fit(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    best_fit = []
    for i in range(y_pred.size):
        for j in range(len(ind)):
            if ind[j][0] == y_pred[i]:
                best_fit.append(ind[j][1])
    return best_fit, ind, w


def cluster_acc(y_true, y_pred):
    _, ind, w = best_cluster_fit(y_true, y_pred)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

#d=np.array(d)
print(np.shape(d))
#y=np.array(y)
print(np.shape(y))


standard_embedding = umap.UMAP(n_neighbors=5).fit_transform(d)
time_start = time.time()
kmeans_labels = cluster.KMeans(n_clusters=10).fit_predict(d)
time_end = time.time()
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=y, s=1, cmap='Spectral');

plt.show();
#print(adjusted_rand_score(d.iloc[:4000, 784], kmeans_labels),adjusted_mutual_info_score(d.iloc[:4000, 784], kmeans_labels))
acc = np.round(cluster_acc(y, kmeans_labels),5)
print("Accuracy: ",acc)
print("NMI: ",normalized_mutual_info_score(y, kmeans_labels))
print('k-means runtime: {} seconds'.format(time_end-time_start))

time_start = time.time()
kmeans_labels = cluster.KMeans(n_clusters=10).fit_predict(standard_embedding)
time_end = time.time()
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=y, s=1, cmap='Spectral');

plt.show();
#print(adjusted_rand_score(d.iloc[:4000, 784], kmeans_labels),adjusted_mutual_info_score(d.iloc[:4000, 784], kmeans_labels))
acc = np.round(cluster_acc(y, kmeans_labels),5)
print("Accuracy: ",acc)
print("NMI: ",normalized_mutual_info_score(y, kmeans_labels))
print('k-means runtime: {} seconds'.format(time_end-time_start))