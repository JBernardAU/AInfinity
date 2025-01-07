import numpy as np

class AdaptiveSimulatedAnnealing:
    def __init__(self, objective_function, bounds, initial_temperature=1000, cooling_rate=0.95, perturbation_scale=0.1, max_iterations=1000):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.perturbation_scale = perturbation_scale
        self.max_iterations = max_iterations
        self.dim = len(bounds)

        # Initialize current solution
        self.current_solution = np.random.uniform(
            low=self.bounds[:, 0], high=self.bounds[:, 1]
        )
        self.current_value = self.objective_function(self.current_solution)
        self.best_solution = self.current_solution
        self.best_value = self.current_value

    def optimize(self):
        temperature = self.initial_temperature
        perturbation_factor = self.perturbation_scale

        for iteration in range(self.max_iterations):
            # Generate a new candidate solution with adaptive perturbation
            perturbation = np.random.uniform(-perturbation_factor, perturbation_factor, self.dim)
            new_solution = self.current_solution + perturbation
            new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
            new_value = self.objective_function(new_solution)

            # Decide whether to accept the new solution
            if new_value < self.current_value or np.random.rand() < np.exp((self.current_value - new_value) / temperature):
                self.current_solution = new_solution
                self.current_value = new_value

                # Update the best solution if the new solution is better
                if new_value < self.best_value:
                    self.best_solution = new_solution
                    self.best_value = new_value
                    # Reduce the perturbation scale when finding a better solution
                    perturbation_factor *= 0.95

            # Cool down the temperature and adapt perturbation scale
            temperature *= self.cooling_rate
            perturbation_factor = max(perturbation_factor, 1e-6)  # Ensure perturbation doesn't vanish

            # Report progress
            print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Value: {self.best_value}, Temperature: {temperature}, Perturbation Scale: {perturbation_factor}")

            # Early stopping if temperature is too low
            if temperature < 1e-8:
                break

        return self.best_solution, self.best_value

# Example usage
def objective_function(x):
    return np.sum(x**2)  # Minimize sum of squares

bounds = [(-10, 10), (-10, 10)]  # Bounds for each dimension
optimizer = AdaptiveSimulatedAnnealing(objective_function, bounds, max_iterations=1000)
best_solution, best_value = optimizer.optimize()
print(f"Best Solution: {best_solution}, Best Value: {best_value}")
