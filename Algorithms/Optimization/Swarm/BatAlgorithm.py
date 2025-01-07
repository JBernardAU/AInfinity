import numpy as np

class BatAlgorithm:
    def __init__(self, objective_function, bounds, num_bats=20, alpha=0.9, gamma=0.9, freq_min=0.0, freq_max=2.0, loudness=1.0, pulse_rate=0.5, max_generations=100):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.num_bats = num_bats
        self.alpha = alpha  # Loudness decrease factor
        self.gamma = gamma  # Pulse rate increase factor
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.max_generations = max_generations
        self.dim = len(bounds)

        # Initialize bat positions, velocities, and frequencies
        self.positions = np.random.uniform(
            low=self.bounds[:, 0], high=self.bounds[:, 1], size=(self.num_bats, self.dim)
        )
        self.velocities = np.zeros((self.num_bats, self.dim))
        self.frequencies = np.zeros(self.num_bats)
        self.fitness = np.array([self.objective_function(pos) for pos in self.positions])

        # Initialize best solution
        self.best_position = self.positions[np.argmin(self.fitness)]
        self.best_fitness = np.min(self.fitness)

    def optimize(self):
        for generation in range(self.max_generations):
            for i in range(self.num_bats):
                # Update frequency, velocity, and position
                self.frequencies[i] = self.freq_min + (self.freq_max - self.freq_min) * np.random.uniform(0, 1)
                self.velocities[i] += (self.positions[i] - self.best_position) * self.frequencies[i]
                new_position = self.positions[i] + self.velocities[i]
                new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

                # Pulse rate-based exploration
                if np.random.uniform(0, 1) > self.pulse_rate:
                    random_position = self.best_position + 0.001 * np.random.normal(0, 1, self.dim)
                    new_position = np.clip(random_position, self.bounds[:, 0], self.bounds[:, 1])

                # Evaluate fitness
                new_fitness = self.objective_function(new_position)

                # Accept the new solution based on loudness
                if new_fitness < self.fitness[i] and np.random.uniform(0, 1) < self.loudness:
                    self.positions[i] = new_position
                    self.fitness[i] = new_fitness

                    # Update the best solution
                    if new_fitness < self.best_fitness:
                        self.best_position = new_position
                        self.best_fitness = new_fitness

            # Update loudness and pulse rate
            self.loudness *= self.alpha
            self.pulse_rate *= (1 - np.exp(-self.gamma * generation))

            # Report progress
            print(f"Generation {generation + 1}/{self.max_generations}, Best Fitness: {self.best_fitness}")

        return self.best_position, self.best_fitness

# Example usage
def objective_function(x):
    return np.sum(x**2)  # Minimize sum of squares

bounds = [(-10, 10), (-10, 10)]  # Bounds for each dimension
optimizer = BatAlgorithm(objective_function, bounds, max_generations=50)
best_solution, best_fitness = optimizer.optimize()
print(f"Best Solution: {best_solution}, Best Fitness: {best_fitness}")
