import numpy as np

class CMAES:
    def __init__(self, objective_function, bounds, population_size=10, sigma=0.5, max_generations=100):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.population_size = population_size
        self.sigma = sigma  # Initial step size
        self.max_generations = max_generations

        # Initialize parameters
        self.mean = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
        self.covariance = np.eye(self.dim)  # Initial covariance matrix
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.covariance)

        # Strategy parameters
        self.weights = np.log(self.population_size + 0.5) - np.log(np.arange(1, self.population_size + 1))
        self.weights /= self.weights.sum()
        self.mu_eff = 1 / np.sum(self.weights**2)
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 3)
        self.d_sigma = 1 + self.c_sigma + 2 * max(np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1, 0)
        self.c_c = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c_1 = 2 / ((self.dim + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dim + 2)**2 + self.mu_eff))

        # Evolution path variables
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)

    def optimize(self):
        for generation in range(self.max_generations):
            # Generate offspring
            samples = np.random.multivariate_normal(self.mean, self.covariance, self.population_size)
            samples = np.clip(samples, self.bounds[:, 0], self.bounds[:, 1])
            fitness = np.array([self.objective_function(ind) for ind in samples])

            # Select the best offspring
            sorted_indices = np.argsort(fitness)
            samples = samples[sorted_indices]
            fitness = fitness[sorted_indices]

            # Update mean
            new_mean = np.sum(self.weights[:, np.newaxis] * samples[:len(self.weights)], axis=0)

            # Update evolution paths
            y = new_mean - self.mean
            z = np.dot(self.eigenvectors.T, y) / self.sigma
            self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * z

            norm_p_sigma = np.linalg.norm(self.p_sigma)
            self.sigma *= np.exp((norm_p_sigma / np.sqrt(self.dim)) - 1) / self.d_sigma

            self.p_c = (1 - self.c_c) * self.p_c + np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y / self.sigma

            # Update covariance matrix
            delta_h_sigma = (1 - (norm_p_sigma / np.sqrt(1 - (1 - self.c_sigma)**(2 * (generation + 1)))))
            rank_one = np.outer(self.p_c, self.p_c)
            rank_mu = np.sum([
                self.weights[i] * np.outer(samples[i] - self.mean, samples[i] - self.mean)
                for i in range(len(self.weights))
            ], axis=0)
            self.covariance = (1 - self.c_1 - self.c_mu) * self.covariance + self.c_1 * rank_one + self.c_mu * rank_mu

            # Update eigenvalues and eigenvectors for sampling
            self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.covariance)

            # Update mean
            self.mean = new_mean

            print(f"Generation {generation + 1}/{self.max_generations}, Best Fitness: {fitness[0]}")

        return samples[0], fitness[0]

# Example usage
def objective_function(x):
    return np.sum(x**2)  # Minimize sum of squares

bounds = [(-5, 5), (-5, 5)]  # Bounds for each dimension
optimizer = CMAES(objective_function, bounds, max_generations=50)
best_solution, best_fitness = optimizer.optimize()
print(f"Best Solution: {best_solution}, Best Fitness: {best_fitness}")
