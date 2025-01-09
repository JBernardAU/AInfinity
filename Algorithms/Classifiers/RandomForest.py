import numpy as np
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

class RandomForest:
    def __init__(self, n_estimators=10, max_features=None, max_depth=None, min_samples_split=2):
        """
        Initialize the Random Forest classifier.

        :param n_estimators: Number of trees in the forest.
        :param max_features: Maximum number of features to consider for splitting a node.
        :param max_depth: Maximum depth of each tree.
        :param min_samples_split: Minimum number of samples required to split a node.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.subsets = []

    def _bootstrap_sample(self, X, y):
        """
        Generate a bootstrap sample from the dataset.

        :param X: Feature matrix.
        :param y: Target labels.
        :return: Bootstrap sample of X and y.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _most_common_label(self, y):
        """
        Return the most common label in y.

        :param y: Array of labels.
        :return: Most common label.
        """
        return Counter(y).most_common(1)[0][0]

    def fit(self, X, y):
        """
        Fit the Random Forest model to the data.

        :param X: Feature matrix.
        :param y: Target labels.
        """
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_features=self.max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict the labels for the given feature matrix.

        :param X: Feature matrix.
        :return: Predicted labels.
        """
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(self._most_common_label, axis=0, arr=tree_predictions)

# Example usage:
if __name__ == "__main__":
    # Generate a simple dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest
    rf = RandomForest(n_estimators=10, max_features=3, max_depth=5)
    rf.fit(X_train, y_train)

    # Predict and evaluate
    predictions = rf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
