import numpy as np

class SimulatedAnnealing:
    def __init__(self, objective_function, bounds, initial_temperature=1000, cooling_rate=0.95, max_iterations=1000):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
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

        for iteration in range(self.max_iterations):
            # Generate a new candidate solution
            new_solution = self.current_solution + np.random.uniform(-0.1, 0.1, self.dim)
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

            # Cool down the temperature
            temperature *= self.cooling_rate

            # Report progress
            print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Value: {self.best_value}, Temperature: {temperature}")

            # Early stopping if temperature is too low
            if temperature < 1e-8:
                break

        return self.best_solution, self.best_value

# Example usage
def objective_function(x):
    return np.sum(x**2)  # Minimize sum of squares

bounds = [(-10, 10), (-10, 10)]  # Bounds for each dimension
optimizer = SimulatedAnnealing(objective_function, bounds, max_iterations=1000)
best_solution, best_value = optimizer.optimize()
print(f"Best Solution: {best_solution}, Best Value: {best_value}")
