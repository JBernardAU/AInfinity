import numpy as np

class DifferentialEvolution:
    def __init__(self, objective_function, bounds, pop_size=20, mutation_factor=0.8, crossover_prob=0.7, max_generations=100):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.max_generations = max_generations
        self.dim = len(bounds)

        # Initialize population
        self.population = np.random.uniform(
            low=self.bounds[:, 0], high=self.bounds[:, 1], size=(self.pop_size, self.dim)
        )
        self.fitness = np.array([self.objective_function(ind) for ind in self.population])

    def optimize(self):
        for generation in range(self.max_generations):
            new_population = []
            for i in range(self.pop_size):
                # Mutation
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                a, b, c = self.population[np.random.choice(candidates, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds[:, 0], self.bounds[:, 1])

                # Crossover
                trial = np.copy(self.population[i])
                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                trial[crossover_mask] = mutant[crossover_mask]

                # Selection
                trial_fitness = self.objective_function(trial)
                if trial_fitness < self.fitness[i]:
                    new_population.append(trial)
                    self.fitness[i] = trial_fitness
                else:
                    new_population.append(self.population[i])

            self.population = np.array(new_population)

            # Report progress
            best_idx = np.argmin(self.fitness)
            print(f"Generation {generation + 1}/{self.max_generations}, Best Fitness: {self.fitness[best_idx]}")

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]

# Example usage
def objective_function(x):
    return np.sum(x**2)  # Minimize sum of squares

bounds = [(-10, 10), (-10, 10)]  # Bounds for each dimension
optimizer = DifferentialEvolution(objective_function, bounds, max_generations=50)
best_solution, best_fitness = optimizer.optimize()
print(f"Best Solution: {best_solution}, Best Fitness: {best_fitness}")
