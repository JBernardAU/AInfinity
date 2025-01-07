import numpy as np
from scipy.stats import norm

class TPEOptimization:
    def __init__(self, objective_function, bounds, n_initial_points=10, max_iterations=50, gamma=0.15):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.n_initial_points = n_initial_points
        self.max_iterations = max_iterations
        self.gamma = gamma  # Fraction of samples in the "good" set
        self.dim = len(bounds)

        # Initialize random samples
        self.X_samples = np.random.uniform(
            low=self.bounds[:, 0], high=self.bounds[:, 1], size=(self.n_initial_points, self.dim)
        )
        self.Y_samples = np.array([self.objective_function(x) for x in self.X_samples])

    def optimize(self):
        for iteration in range(self.max_iterations):
            # Partition samples into good and bad sets
            threshold = np.percentile(self.Y_samples, self.gamma * 100)
            good_indices = self.Y_samples <= threshold
            bad_indices = self.Y_samples > threshold

            X_good = self.X_samples[good_indices]
            X_bad = self.X_samples[bad_indices]

            # Fit Parzen estimators
            good_means, good_stds = self._fit_parzen_estimator(X_good)
            bad_means, bad_stds = self._fit_parzen_estimator(X_bad)

            # Propose a new sample
            next_sample = self._propose_location(good_means, good_stds, bad_means, bad_stds)

            # Evaluate the objective function
            next_value = self.objective_function(next_sample)

            # Add the new sample to the dataset
            self.X_samples = np.vstack((self.X_samples, next_sample))
            self.Y_samples = np.append(self.Y_samples, next_value)

            # Report progress
            best_idx = np.argmin(self.Y_samples)
            print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Value: {self.Y_samples[best_idx]}")

        best_idx = np.argmin(self.Y_samples)
        return self.X_samples[best_idx], self.Y_samples[best_idx]

    def _fit_parzen_estimator(self, X):
        """Fit a Parzen estimator by calculating the mean and standard deviation for each dimension."""
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0) + 1e-6  # Avoid zero standard deviation
        return means, stds

    def _propose_location(self, good_means, good_stds, bad_means, bad_stds):
        """Propose a new sample based on the ratio of good to bad densities."""
        best_sample = None
        best_ratio = float('-inf')

        for _ in range(100):  # Random search for simplicity
            candidate = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
            p_good = np.prod(norm.pdf(candidate, loc=good_means, scale=good_stds))
            p_bad = np.prod(norm.pdf(candidate, loc=bad_means, scale=bad_stds))

            ratio = p_good / (p_bad + 1e-6)  # Avoid division by zero
            if ratio > best_ratio:
                best_sample = candidate
                best_ratio = ratio

        return best_sample

# Example usage
def objective_function(x):
    return np.sum(x**2)  # Minimize sum of squares

bounds = [(-10, 10), (-10, 10)]  # Bounds for each dimension
optimizer = TPEOptimization(objective_function, bounds, max_iterations=50)
best_solution, best_value = optimizer.optimize()
print(f"Best Solution: {best_solution}, Best Value: {best_value}")
