import numpy as np
from collections import Counter


class KNNClassifier:
    def __init__(self, k=3):
        """
        Initializes the k-Nearest Neighbors classifier.

        Args:
            k (int): Number of neighbors to consider.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fits the classifier with training data.

        Args:
            X (np.ndarray): Training data features (shape: [n_samples, n_features]).
            y (np.ndarray): Training data labels (shape: [n_samples]).
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts the labels for the given test data.

        Args:
            X (np.ndarray): Test data features (shape: [n_samples, n_features]).

        Returns:
            np.ndarray: Predicted labels (shape: [n_samples]).
        """
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        """
        Predicts the label for a single test sample.

        Args:
            x (np.ndarray): A single test sample (shape: [n_features]).

        Returns:
            int: Predicted label.
        """
        # Calculate distances to all training samples
        distances = np.linalg.norm(self.X_train - x, axis=1)

        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# Example usage
if __name__ == "__main__":
    # Example dataset (2D points with two classes)
    X_train = np.array([[1, 2], [2, 3], [3, 3], [6, 6], [7, 8], [8, 8]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([[2, 2], [5, 5]])

    # Initialize and train the kNN classifier
    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)

    # Predict labels for test data
    predictions = knn.predict(X_test)
    print("Predictions:", predictions)
