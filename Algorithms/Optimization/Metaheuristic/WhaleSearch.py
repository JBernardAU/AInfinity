import numpy as np
import random


class WhaleOptimizationAlgorithm:
    """
    Implementation of the Whale Optimization Algorithm (WOA) for optimization problems.
    """

    def __init__(self,
                 objective_function,
                 variable_bounds,
                 population_size=30,
                 max_iterations=100,
                 seed=None):
        """
        Initialize the Whale Optimization Algorithm.

        Parameters:
            objective_function (callable): The function to minimize.
            variable_bounds (list[tuple]): Bounds for each variable (lower, upper).
            population_size (int): Number of whales in the population.
            max_iterations (int): Maximum number of iterations.
            seed (int, optional): Random seed for reproducibility.
        """
        self.objective_function = objective_function
        self.variable_bounds = np.array(variable_bounds)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.num_variables = len(variable_bounds)
        self.seed = seed

        # Random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Initialize population
        self.population = np.random.uniform(
            self.variable_bounds[:, 0],
            self.variable_bounds[:, 1],
            (self.population_size, self.num_variables)
        )
        self.best_solution = None
        self.best_fitness = np.inf

    def _update_population(self, a):
        """
        Update the population based on the encircling, bubble-net, and search strategies.
        """
        for i in range(self.population_size):
            whale = self.population[i]
            r1 = np.random.uniform(0, 1)  # Random numbers for strategy control
            r2 = np.random.uniform(0, 1)

            A = 2 * a * r1 - a  # Equation (2.3)
            C = 2 * r2  # Equation (2.4)

            if np.random.uniform() < 0.5:  # Exploitation phase
                if abs(A) < 1:
                    # Shrinking encircling mechanism
                    D = abs(C * self.best_solution - whale)
                    new_position = self.best_solution - A * D
                else:
                    # Spiral updating position
                    b = 1  # Spiral shape constant
                    l = np.random.uniform(-1, 1)  # Random step size
                    D_prime = abs(self.best_solution - whale)
                    new_position = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_solution
            else:  # Exploration phase
                random_whale = self.population[np.random.randint(self.population_size)]
                D = abs(C * random_whale - whale)
                new_position = random_whale - A * D

            # Ensure new position is within bounds
            new_position = np.clip(new_position, self.variable_bounds[:, 0], self.variable_bounds[:, 1])

            # Update whale's position
            self.population[i] = new_position

    def run(self):
        """
        Run the Whale Optimization Algorithm.

        Returns:
            tuple: Best solution and its objective function value.
        """
        for iteration in range(self.max_iterations):
            # Evaluate fitness for all whales
            fitness = np.array([self.objective_function(whale) for whale in self.population])

            # Update the best solution
            min_fitness = np.min(fitness)
            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                self.best_solution = self.population[np.argmin(fitness)].copy()

            # Update population
            a = 2 - iteration * (2 / self.max_iterations)  # Linearly decreasing from 2 to 0
            self._update_population(a)

            # (Optional) Uncomment for debug/progress tracking
            # print(f"Iteration {iteration+1}, Best Fitness: {self.best_fitness}")

        return self.best_solution, self.best_fitness


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    # Example: Minimize the Sphere function
    def sphere_function(x):
        return sum(xi ** 2 for xi in x)


    # Bounds for each variable
    variable_bounds = [(-10, 10)] * 5  # 5-dimensional Sphere function

    # Instantiate the WOA optimizer
    woa = WhaleOptimizationAlgorithm(
        objective_function=sphere_function,
        variable_bounds=variable_bounds,
        population_size=30,
        max_iterations=100,
        seed=42
    )

    # Run the optimizer
    best_solution, best_fitness = woa.run()
    print("Best Solution:", best_solution)
    print("Best Objective Value:", best_fitness)
