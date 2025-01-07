import numpy as np

class FireflyAlgorithm:
    def __init__(self, objective_function, bounds, num_fireflies=20, alpha=0.5, beta=1.0, gamma=1.0, max_generations=100):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.num_fireflies = num_fireflies
        self.alpha = alpha  # Randomness strength
        self.beta = beta  # Attraction coefficient
        self.gamma = gamma  # Light absorption coefficient
        self.max_generations = max_generations
        self.dim = len(bounds)

        # Initialize firefly positions randomly within bounds
        self.positions = np.random.uniform(
            low=self.bounds[:, 0], high=self.bounds[:, 1], size=(self.num_fireflies, self.dim)
        )
        self.fitness = np.array([self.objective_function(pos) for pos in self.positions])

    def optimize(self):
        for generation in range(self.max_generations):
            for i in range(self.num_fireflies):
                for j in range(self.num_fireflies):
                    if self.fitness[j] < self.fitness[i]:  # Move firefly i toward firefly j
                        distance = np.linalg.norm(self.positions[i] - self.positions[j])
                        attractiveness = self.beta * np.exp(-self.gamma * distance**2)
                        random_factor = self.alpha * (np.random.uniform(-0.5, 0.5, self.dim))

                        self.positions[i] += attractiveness * (self.positions[j] - self.positions[i]) + random_factor
                        self.positions[i] = np.clip(self.positions[i], self.bounds[:, 0], self.bounds[:, 1])

                # Update fitness
                self.fitness[i] = self.objective_function(self.positions[i])

            # Report progress
            best_idx = np.argmin(self.fitness)
            print(f"Generation {generation + 1}/{self.max_generations}, Best Fitness: {self.fitness[best_idx]}")

        best_idx = np.argmin(self.fitness)
        return self.positions[best_idx], self.fitness[best_idx]

# Example usage
def objective_function(x):
    return np.sum(x**2)  # Minimize sum of squares

bounds = [(-10, 10), (-10, 10)]  # Bounds for each dimension
optimizer = FireflyAlgorithm(objective_function, bounds, max_generations=50)
best_solution, best_fitness = optimizer.optimize()
print(f"Best Solution: {best_solution}, Best Fitness: {best_fitness}")
