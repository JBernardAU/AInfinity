import numpy as np

class KMeansClassifier:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        """
        Parameters:
        - n_clusters (int): Number of clusters to form.
        - max_iter (int): Maximum number of iterations for the algorithm.
        - tol (float): Tolerance for convergence. Stops if the centroids do not move more than this value.
        - random_state (int, optional): Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        """Compute k-means clustering.

        Parameters:
        - X (array-like): Shape (n_samples, n_features). Data points to cluster.
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # Initialize centroids randomly from the data points
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iter):
            # Assign clusters based on the nearest centroid
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)

            # Compute new centroids as the mean of assigned points
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])

            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

        self.labels_ = labels

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters:
        - X (array-like): Shape (n_samples, n_features). Data points to predict.

        Returns:
        - labels (array): Index of the cluster each sample belongs to.
        """
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def _compute_distances(self, X):
        """Compute the distance between each data point and each centroid."""
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # Generate synthetic data
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

    # Train K-Means classifier
    kmeans = KMeansClassifier(n_clusters=3, random_state=42)
    kmeans.fit(X)

    # Predict labels
    labels = kmeans.predict(X)

    # Plot the results
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
    plt.legend()
    plt.show()
