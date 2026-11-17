import sys
import numpy as np
from sklearn.metrics import silhouette_score
import symnmfmodule
import symnmf

np.random.seed(1234)

"""
Calculates the straight-line distance between two points.
Parameter point1: First d-dimensional point.
Parameter point2: Second d-dimensional point.
Return: The square root of the distance sum.   """
def euclidian_distance(point1, point2):
    distance=0
    for i in range(len(point1)):
        distance = distance + (point1[i] - point2[i])**2
    return distance**0.5

"""
Iterates through all clusters to calculate their centers.
Parameter clusters: List of points for each cluster.
Return: A list of new centroid coordinates.
"""
def Update_Allcentroid(clusters):
    new_centroids = []
    for i in range(len(clusters)):
        new_centroids.append(Update_centroid(clusters[i]))
    return new_centroids

"""
Calculates the centroid of a single cluster of points.
Parameter cluster: A list of points assigned to the cluster
Return: A list representing the new centroid, or an empty list if the cluster is empty
"""
def Update_centroid(cluster):
    new_centroid = []
    if len(cluster) == 0:
        return []
    d = len(cluster[0])
    for i in range (d):
        sum_coordinate = 0
        for j in range(len(cluster)):
            sum_coordinate += cluster[j][i]
        new_centroid.append(sum_coordinate/len(cluster))
    return new_centroid

"""
Checks if centroids moved less than a defined threshold.    
Parameter old_centroid: Previous centroids list.
Parameter new_centroid: Current centroids list.
Parameter eps: Convergence threshold (epsilon).
Return: True if converged, False otherwise.
    """
def centroid_distance(old_centroid, new_centroid, eps):
    i = 0
    while i < len(old_centroid):
        dis = euclidian_distance(old_centroid[i], new_centroid[i])
        if dis > eps:
            return False
        i += 1
    return True

"""
Assigns points to clusters and tracks labels to avoid re-calculation.
Parameter points: List of data points.
Parameter centroids: Current list of cluster centers.
Parameter k: Number of clusters.
Return: Clusters list, labels list, and index of point[0]'s cluster.
"""
def assign_points(points, centroids, k):
    clusters = [[] for _ in range(k)]
    labels = [0] * len(points)
    index_p1 = -1
    for i, point in enumerate(points):
        min_dist = float('inf')
        min_centroid = -1
        for j in range(k):
            dist = euclidian_distance(point, centroids[j])
            if dist < min_dist:
                min_dist = dist
                min_centroid = j
        labels[i] = min_centroid
        clusters[min_centroid].append(point)
        if i == 0:
            index_p1 = min_centroid
    return clusters, labels, index_p1

"""
Moves the first point to an empty cluster if needed.
Parameter clusters: List of current clusters.
Parameter points: Original dataset.
Parameter labels: List of cluster assignments.
Parameter index_p1: Current cluster index of points[0].
Return: None (modifies lists in place).
"""
def handle_empty(clusters, points, labels, index_p1):
    for i in range(len(clusters)):
        if not clusters[i]: 
            if i != index_p1:
                clusters[i].append(points[0])
                labels[0] = i 
                for j, p in enumerate(clusters[index_p1]):
                    if p == points[0]:
                        clusters[index_p1].pop(j)
                        break
            break
"""
Clusters data points into k groups using the K-means logic.
Parameter points: Input dataset for clustering.
Parameter k: Number of clusters to create.
Parameter max_iter: Maximum iteration limit.
Parameter eps: Convergence threshold value.
Return: List of cluster labels for each point.
"""
def run_kmeans(points, k, max_iter=300, eps=1e-4):
    centroids = [points[i] for i in range(k)]
    final_labels = [0] * len(points)
    
    for i in range(max_iter):
        clusters, current_labels, index_p1 = assign_points(points, centroids, k)
        handle_empty(clusters, points, current_labels, index_p1)
        new_centroids = Update_Allcentroid(clusters)
        final_labels = current_labels
        converged = centroid_distance(centroids, new_centroids, eps)
        centroids = new_centroids
        if converged:
            break       
    return final_labels


"""
    Executes the comparison between SymNMF and K-means:
    1. Validates command-line arguments and k constraints.
    2. Loads data and runs both clustering algorithms.
    3. Calculates and prints Silhouette scores formatted to 4 decimal places.
"""
def main():
    try:
        # Validate number of arguments
        if len(sys.argv) != 3:
            print("An Error Has Occurred")
            sys.exit(1)
        k = int(sys.argv[1])
        filename = sys.argv[2]
        if k < 1:
            print("An Error Has Occurred")
            sys.exit(1)
        data = np.genfromtxt(filename, delimiter=',')
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        N, d = data.shape
        if k >= N:
            print("An Error Has Occurred")
            sys.exit(1)
        data_list = data.tolist()
        # SymNMF Analysis: Get H from C extension and find labels using argmax
        final_H=np.array(symnmf.run_symnmf(data_list,N,k))
        nmf_clusters = np.argmax(final_H, axis=1)
        # K-means Analysis: Run manual implementation from HW1
        kmeans_clusters=run_kmeans(data_list,k)
        # Calculate Silhouette scores
        nmf_score = silhouette_score(data, nmf_clusters)
        kmeans_score = silhouette_score(data, kmeans_clusters)
        print(f"nmf: {nmf_score:.4f}")
        print(f"kmeans: {kmeans_score:.4f}")

    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()

