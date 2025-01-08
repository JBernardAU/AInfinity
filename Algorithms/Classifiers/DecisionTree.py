import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        """
        Initializes the Decision Tree classifier.

        Args:
            max_depth (int, optional): The maximum depth of the tree. Default is None (no limit).
        """
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Fits the classifier to the training data.

        Args:
            X (np.ndarray): Training data features (shape: [n_samples, n_features]).
            y (np.ndarray): Training data labels (shape: [n_samples]).
        """
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        """
        Predicts the labels for the given test data.

        Args:
            X (np.ndarray): Test data features (shape: [n_samples, n_features]).

        Returns:
            np.ndarray: Predicted labels (shape: [n_samples]).
        """
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth=0):
        """
        Recursively builds the decision tree.

        Args:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.
            depth (int): Current depth of the tree.

        Returns:
            dict: A dictionary representation of the decision tree.
        """
        # Check stopping conditions
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        if num_classes == 1 or num_samples == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return self._create_leaf(y)

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return self._create_leaf(y)

        # Split the data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_subtree,
            "right": right_subtree
        }

    def _create_leaf(self, y):
        """
        Creates a leaf node.

        Args:
            y (np.ndarray): Labels of the data in the leaf.

        Returns:
            dict: A dictionary representing a leaf node.
        """
        most_common_label = np.bincount(y).argmax()
        return {"label": most_common_label}

    def _find_best_split(self, X, y):
        """
        Finds the best feature and threshold to split the data.

        Args:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.

        Returns:
            tuple: Best feature index and best threshold value.
        """
        num_samples, num_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        """
        Calculates the information gain of a split.

        Args:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.
            feature (int): Feature index to split on.
            threshold (float): Threshold value to split on.

        Returns:
            float: Information gain of the split.
        """
        parent_entropy = self._entropy(y)

        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return 0

        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])

        left_weight = np.sum(left_indices) / len(y)
        right_weight = np.sum(right_indices) / len(y)

        child_entropy = left_weight * left_entropy + right_weight * right_entropy

        return parent_entropy - child_entropy

    def _entropy(self, y):
        """
        Calculates the entropy of a set of labels.

        Args:
            y (np.ndarray): Labels.

        Returns:
            float: Entropy of the labels.
        """
        hist = np.bincount(y)
        probabilities = hist / len(y)
        probabilities = probabilities[probabilities > 0]  # Avoid log(0)
        return -np.sum(probabilities * np.log2(probabilities))

    def _predict_single(self, x, tree):
        """
        Predicts the label for a single test sample.

        Args:
            x (np.ndarray): A single test sample.
            tree (dict): Current subtree or leaf.

        Returns:
            int: Predicted label.
        """
        if "label" in tree:
            return tree["label"]

        feature = tree["feature"]
        threshold = tree["threshold"]

        if x[feature] <= threshold:
            return self._predict_single(x, tree["left"])
        else:
            return self._predict_single(x, tree["right"])

# Example usage
if __name__ == "__main__":
    # Example dataset (2D points with binary labels)
    X_train = np.array([[2, 3], [1, 1], [3, 2], [6, 5], [7, 8], [8, 9]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([[2, 2], [5, 6]])

    # Initialize and train the decision tree classifier
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X_train, y_train)

    # Predict labels for test data
    predictions = tree.predict(X_test)
    print("Predictions:", predictions)
