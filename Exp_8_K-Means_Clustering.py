#Algorithm
#S-1 : Select the number of clusters (K) and initialize K centroids randomly.
#S-2 : Assign each data point to the nearest centroid.
#S-3 : Compute new centroids as the mean of all points in each cluster.
#S-4 : Repeat steps 2 and 3 until centroids do not change significantly.
#S-5 : Return the final cluster assignments and centroids.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
k = 3 
kmeans = KMeans(n_clusters=k, random_state=100, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis', edgecolors='k', alpha=0.9)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='red', marker='X', label='Centroids')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering on Iris Dataset")
plt.legend()
plt.show()
