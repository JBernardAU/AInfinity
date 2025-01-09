import numpy as np

class Kernel:
    """
    Base class for kernel functions. Extend this class to implement custom kernels.
    """
    def __call__(self, x1, x2):
        raise NotImplementedError("Kernel must implement __call__ method.")

class LinearKernel(Kernel):
    def __call__(self, x1, x2):
        return np.dot(x1, x2)

class PolynomialKernel(Kernel):
    def __init__(self, degree=3, coef0=1):
        self.degree = degree
        self.coef0 = coef0

    def __call__(self, x1, x2):
        return (np.dot(x1, x2) + self.coef0) ** self.degree

class RBFKernel(Kernel):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def __call__(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

class SVM:
    def __init__(self, kernel=None, C=1.0, n_iters=1000):
        """
        Initialize the Support Vector Machine.

        :param kernel: Kernel object (e.g., LinearKernel, PolynomialKernel, RBFKernel).
        :param C: Regularization parameter (controls trade-off between maximizing margin and minimizing slack).
        :param n_iters: Number of iterations for optimization.
        """
        self.kernel = kernel if kernel is not None else LinearKernel()
        self.C = C
        self.n_iters = n_iters
        self.alpha = None
        self.b = 0
        self.support_vectors = None
        self.support_labels = None

    def fit(self, X, y):
        """
        Train the SVM model using the dual form of the optimization problem.

        :param X: Feature matrix (shape: [n_samples, n_features]).
        :param y: Target labels (shape: [n_samples], must be -1 or 1).
        """
        n_samples, n_features = X.shape

        # Ensure labels are -1 or 1
        y = np.where(y <= 0, -1, 1)

        # Initialize alpha values
        self.alpha = np.zeros(n_samples)

        # Kernel matrix
        K = np.array([[self.kernel(X[i], X[j]) for j in range(n_samples)] for i in range(n_samples)])

        # Gradient descent to solve dual problem
        for _ in range(self.n_iters):
            for i in range(n_samples):
                # Compute margin for sample i
                margin = np.sum(self.alpha * y * K[:, i]) + self.b
                # Compute gradient and update alpha
                if y[i] * margin < 1:
                    self.alpha[i] += self.C * (1 - y[i] * margin)

        # Identify support vectors
        support_vector_indices = self.alpha > 1e-5
        self.support_vectors = X[support_vector_indices]
        self.support_labels = y[support_vector_indices]
        self.alpha = self.alpha[support_vector_indices]

        # Compute bias
        self.b = np.mean(
            [y_k - np.sum(self.alpha * self.support_labels * K[i, support_vector_indices])
             for i, y_k in enumerate(y[support_vector_indices])]
        )

    def predict(self, X):
        """
        Predict labels for the given feature matrix.

        :param X: Feature matrix (shape: [n_samples, n_features]).
        :return: Predicted labels (1 or -1).
        """
        predictions = []
        for x in X:
            margin = np.sum(
                self.alpha * self.support_labels * np.array([self.kernel(sv, x) for sv in self.support_vectors])
            ) + self.b
            predictions.append(np.sign(margin))
        return np.array(predictions)

# Example usage
if __name__ == "__main__":
    # Generate a simple non-linear dataset
    from sklearn.datasets import make_circles
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = make_circles(n_samples=100, factor=0.5, noise=0.1, random_state=42)
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the SVM with RBF kernel
    rbf_kernel = RBFKernel(gamma=0.5)
    svm = SVM(kernel=rbf_kernel, C=1.0, n_iters=500)
    svm.fit(X_train, y_train)

    # Predict and evaluate
    predictions = svm.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
