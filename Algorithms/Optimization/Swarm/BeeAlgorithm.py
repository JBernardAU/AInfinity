import numpy as np

class BeeAlgorithm:
    def __init__(self, objective_function, bounds, num_bees=50, elite_bees=10, best_sites=5, neighborhood_size=0.1, max_generations=100):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.num_bees = num_bees
        self.elite_bees = elite_bees
        self.best_sites = best_sites
        self.neighborhood_size = neighborhood_size
        self.max_generations = max_generations
        self.dim = len(bounds)

        # Initialize bee positions randomly within bounds
        self.positions = np.random.uniform(
            low=self.bounds[:, 0], high=self.bounds[:, 1], size=(self.num_bees, self.dim)
        )
        self.fitness = np.array([self.objective_function(pos) for pos in self.positions])

    def optimize(self):
        for generation in range(self.max_generations):
            # Sort bees by fitness
            sorted_indices = np.argsort(self.fitness)
            self.positions = self.positions[sorted_indices]
            self.fitness = self.fitness[sorted_indices]

            # Perform neighborhood search for elite and best sites
            new_positions = []
            for i in range(self.elite_bees):
                new_positions.extend(self._neighborhood_search(self.positions[i], self.elite_bees))
            for i in range(self.elite_bees, self.best_sites):
                new_positions.extend(self._neighborhood_search(self.positions[i], self.best_sites))

            # Random exploration for remaining bees
            remaining_bees = self.num_bees - len(new_positions)
            random_positions = np.random.uniform(
                low=self.bounds[:, 0], high=self.bounds[:, 1], size=(remaining_bees, self.dim)
            )
            new_positions.extend(random_positions)

            # Evaluate fitness of new positions
            new_positions = np.array(new_positions)
            new_fitness = np.array([self.objective_function(pos) for pos in new_positions])

            # Update positions and fitness
            self.positions = new_positions
            self.fitness = new_fitness

            # Report progress
            print(f"Generation {generation + 1}/{self.max_generations}, Best Fitness: {self.fitness[0]}")

        best_index = np.argmin(self.fitness)
        return self.positions[best_index], self.fitness[best_index]

    def _neighborhood_search(self, position, num_neighbors):
        neighbors = []
        for _ in range(num_neighbors):
            neighbor = position + np.random.uniform(
                -self.neighborhood_size, self.neighborhood_size, size=self.dim
            )
            neighbor = np.clip(neighbor, self.bounds[:, 0], self.bounds[:, 1])
            neighbors.append(neighbor)
        return neighbors

# Example usage
def objective_function(x):
    return np.sum(x**2)  # Minimize sum of squares

bounds = [(-10, 10), (-10, 10)]  # Bounds for each dimension
optimizer = BeeAlgorithm(objective_function, bounds, max_generations=50)
best_solution, best_fitness = optimizer.optimize()
print(f"Best Solution: {best_solution}, Best Fitness: {best_fitness}")
