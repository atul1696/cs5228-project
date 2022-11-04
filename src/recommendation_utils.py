import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

def get_euclidean_distance(row, X):
    distances = euclidean_distances(row, X).flatten()
    return distances/distances.sum()

def get_recommendation_weights(row, X, distance_metric='euclidean'):
    if distance_metric=='euclidean':
        distances = get_euclidean_distance(row, X)

    weights = np.exp(-distances)/np.exp(-distances).sum()
    weights = weights/weights.sum()

    return weights
