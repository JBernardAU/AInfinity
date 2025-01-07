import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class GaussianProcessOptimization:
    def __init__(self, objective_function, bounds, kernel=None, n_initial_points=5, max_iterations=50):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.n_initial_points = n_initial_points
        self.max_iterations = max_iterations
        self.dim = len(bounds)

        # Define Gaussian Process kernel
        self.kernel = kernel if kernel else Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-6, normalize_y=True)

        # Initial random samples
        self.X_samples = np.random.uniform(
            low=self.bounds[:, 0], high=self.bounds[:, 1], size=(self.n_initial_points, self.dim)
        )
        self.Y_samples = np.array([self.objective_function(x) for x in self.X_samples])

    def acquisition_function(self, x):
        """Expected Improvement acquisition function."""
        mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
        sigma = np.maximum(sigma, 1e-8)  # Avoid division by zero

        best_y = np.min(self.Y_samples)
        improvement = best_y - mu
        z = improvement / sigma
        ei = improvement * self._cdf(z) + sigma * self._pdf(z)
        return -ei  # Minimize acquisition function

    def _pdf(self, z):
        return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)

    def _cdf(self, z):
        return 0.5 * (1 + np.erf(z / np.sqrt(2)))

    def optimize(self):
        for iteration in range(self.max_iterations):
            # Fit the Gaussian Process model
            self.gp.fit(self.X_samples, self.Y_samples)

            # Find the next point to sample by optimizing the acquisition function
            next_sample = self._propose_location()

            # Evaluate the objective function at the proposed point
            next_value = self.objective_function(next_sample)

            # Add the new sample to the dataset
            self.X_samples = np.vstack((self.X_samples, next_sample))
            self.Y_samples = np.append(self.Y_samples, next_value)

            # Report progress
            best_idx = np.argmin(self.Y_samples)
            print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Value: {self.Y_samples[best_idx]}")

        best_idx = np.argmin(self.Y_samples)
        return self.X_samples[best_idx], self.Y_samples[best_idx]

    def _propose_location(self):
        """Optimize the acquisition function to propose the next sampling location."""
        best_x = None
        best_ei = float('inf')

        for _ in range(100):  # Random search for simplicity
            candidate_x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
            ei = self.acquisition_function(candidate_x)
            if ei < best_ei:
                best_x = candidate_x
                best_ei = ei

        return best_x

# Example usage
def objective_function(x):
    return np.sum(x**2)  # Minimize sum of squares

bounds = [(-10, 10), (-10, 10)]  # Bounds for each dimension
optimizer = GaussianProcessOptimization(objective_function, bounds, max_iterations=50)
best_solution, best_value = optimizer.optimize()
print(f"Best Solution: {best_solution}, Best Value: {best_value}")
