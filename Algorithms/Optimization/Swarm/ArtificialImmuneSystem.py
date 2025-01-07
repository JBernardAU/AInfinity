import numpy as np

class ArtificialImmuneSystem:
    def __init__(self, objective_function, bounds, population_size=50, cloning_rate=0.1, mutation_rate=0.2, max_generations=100):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.cloning_rate = cloning_rate  # Proportion of population to clone
        self.mutation_rate = mutation_rate  # Probability of mutation per clone
        self.max_generations = max_generations
        self.dim = len(bounds)

        # Initialize population
        self.population = np.random.uniform(
            low=self.bounds[:, 0], high=self.bounds[:, 1], size=(self.population_size, self.dim)
        )
        self.fitness = np.array([self.objective_function(ind) for ind in self.population])

    def optimize(self):
        for generation in range(self.max_generations):
            # Select the top candidates based on fitness
            sorted_indices = np.argsort(self.fitness)
            self.population = self.population[sorted_indices]
            self.fitness = self.fitness[sorted_indices]

            # Clone the best individuals
            num_clones = int(self.cloning_rate * self.population_size)
            clones = []
            for i in range(num_clones):
                clone = self.population[i].copy()

                # Mutate the clones
                for d in range(self.dim):
                    if np.random.rand() < self.mutation_rate:
                        clone[d] += np.random.normal(0, 0.1 * (self.bounds[d, 1] - self.bounds[d, 0]))
                clone = np.clip(clone, self.bounds[:, 0], self.bounds[:, 1])
                clones.append(clone)

            # Evaluate the clones
            clones = np.array(clones)
            clone_fitness = np.array([self.objective_function(ind) for ind in clones])

            # Combine population with clones and reselect the top individuals
            combined_population = np.vstack((self.population, clones))
            combined_fitness = np.hstack((self.fitness, clone_fitness))

            sorted_indices = np.argsort(combined_fitness)
            self.population = combined_population[sorted_indices][:self.population_size]
            self.fitness = combined_fitness[sorted_indices][:self.population_size]

            # Report progress
            print(f"Generation {generation + 1}/{self.max_generations}, Best Fitness: {self.fitness[0]}")

        return self.population[0], self.fitness[0]

# Example usage
def objective_function(x):
    return np.sum(x**2)  # Minimize sum of squares

bounds = [(-10, 10), (-10, 10)]  # Bounds for each dimension
optimizer = ArtificialImmuneSystem(objective_function, bounds, max_generations=50)
best_solution, best_fitness = optimizer.optimize()
print(f"Best Solution: {best_solution}, Best Fitness: {best_fitness}")
