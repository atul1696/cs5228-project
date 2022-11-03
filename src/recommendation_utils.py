from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors

def top_k_cosine_distance(row, X, k=10):
    distances = cosine_distances(row, X).flatten()

    return distances.argsort()[:k]

def top_k_nearest_neighbors(row, X, k=10):
    nearest_neighbors = NearestNeighbors(n_neighbors=k).fit(X)
    distances, index_list = nearest_neighbors.kneighbors(row)

    return index_list.flatten()
