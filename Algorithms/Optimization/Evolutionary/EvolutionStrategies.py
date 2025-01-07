import numpy as np

class EvolutionStrategies:
    def __init__(self, objective_function, bounds, population_size=20, sigma=0.1, learning_rate=0.1, max_generations=100):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.sigma = sigma  # Standard deviation for mutation
        self.learning_rate = learning_rate
        self.max_generations = max_generations
        self.dim = len(bounds)

        # Initialize parent solution randomly within bounds
        self.parent = np.random.uniform(
            low=self.bounds[:, 0], high=self.bounds[:, 1], size=self.dim
        )
        self.parent_fitness = self.objective_function(self.parent)

    def optimize(self):
        for generation in range(self.max_generations):
            # Generate offspring solutions
            offspring = []
            for _ in range(self.population_size):
                mutation = np.random.normal(0, self.sigma, self.dim)
                child = np.clip(self.parent + mutation, self.bounds[:, 0], self.bounds[:, 1])
                fitness = self.objective_function(child)
                offspring.append((child, fitness))

            # Sort offspring by fitness (minimization problem)
            offspring.sort(key=lambda x: x[1])

            # Update parent solution using the best offspring
            top_offspring = [o[0] for o in offspring[:self.population_size // 2]]  # Top 50%
            self.parent += self.learning_rate * np.mean(top_offspring - self.parent, axis=0)
            self.parent = np.clip(self.parent, self.bounds[:, 0], self.bounds[:, 1])

            # Update parent fitness
            self.parent_fitness = self.objective_function(self.parent)

            print(f"Generation {generation + 1}/{self.max_generations}, Best Fitness: {self.parent_fitness}")

        return self.parent, self.parent_fitness

# Example usage
def objective_function(x):
    return np.sum(x**2)  # Minimize sum of squares

bounds = [(-10, 10), (-10, 10)]  # Bounds for each dimension
optimizer = EvolutionStrategies(objective_function, bounds, max_generations=50)
best_solution, best_fitness = optimizer.optimize()
print(f"Best Solution: {best_solution}, Best Fitness: {best_fitness}")
