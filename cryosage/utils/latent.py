"""PCA trajectories and nearest observed latent representatives."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def get_nearest_point(data, query):
    data, query = np.asarray(data), np.atleast_2d(query)
    indices = np.argmin(cdist(query, data), axis=1)
    return data[indices], indices


def cluster_kmeans(z, K, on_data=True, reorder=True):
    data = np.asarray(z)
    fitted = KMeans(n_clusters=K, n_init=10, random_state=0, max_iter=10).fit(data)
    centers, labels = fitted.cluster_centers_, fitted.labels_
    if on_data:
        centers, _ = get_nearest_point(data, centers)
    if reorder and len(centers) > 1:
        order = leaves_list(linkage(centers, method="average", metric="euclidean"))
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        centers, labels = centers[order], inverse[labels]
    return labels, centers


def sample_along_pca(z, pca_dim=1, num=4):
    """Sample a PC's 5th--95th percentile interval and return observed neighbors."""
    data = np.asarray(z)
    if data.ndim != 2 or not 1 <= pca_dim <= min(data.shape) or num < 1:
        raise ValueError("Invalid latent shape, PCA component or sample count")
    pca = PCA(n_components=min(data.shape)).fit(data)
    scores = pca.transform(data)
    targets = np.zeros((num, scores.shape[1]), dtype=scores.dtype)
    low, high = np.percentile(scores[:, pca_dim-1], [5, 95])
    targets[:, pca_dim-1] = np.linspace(low, high, num)
    return get_nearest_point(data, pca.inverse_transform(targets))
