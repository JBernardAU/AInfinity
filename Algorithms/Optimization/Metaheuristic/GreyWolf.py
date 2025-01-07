import numpy as np


class GreyWolfOptimizer:
    """
    Implementation of the Grey Wolf Optimizer (GWO) for optimization problems.
    """

    def __init__(self,
                 objective_function,
                 variable_bounds,
                 population_size=30,
                 max_iterations=100,
                 seed=None):
        """
        Initialize the Grey Wolf Optimizer.

        Parameters:
            objective_function (callable): The function to minimize.
            variable_bounds (list[tuple]): Bounds for each variable (lower, upper).
            population_size (int): Number of wolves in the population.
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

        # Initialize wolf population
        self.population = np.random.uniform(
            self.variable_bounds[:, 0],
            self.variable_bounds[:, 1],
            (self.population_size, self.num_variables)
        )
        self.fitness = np.full(self.population_size, np.inf)

        # Initialize alpha, beta, and delta wolves
        self.alpha_position = None
        self.beta_position = None
        self.delta_position = None
        self.alpha_fitness = np.inf
        self.beta_fitness = np.inf
        self.delta_fitness = np.inf

    def _update_leadership(self):
        """
        Update the positions and fitness of alpha, beta, and delta wolves.
        """
        for i in range(self.population_size):
            fit = self.objective_function(self.population[i])
            self.fitness[i] = fit

            # Update alpha, beta, and delta
            if fit < self.alpha_fitness:
                self.delta_fitness = self.beta_fitness
                self.delta_position = self.beta_position
                self.beta_fitness = self.alpha_fitness
                self.beta_position = self.alpha_position
                self.alpha_fitness = fit
                self.alpha_position = self.population[i].copy()
            elif fit < self.beta_fitness:
                self.delta_fitness = self.beta_fitness
                self.delta_position = self.beta_position
                self.beta_fitness = fit
                self.beta_position = self.population[i].copy()
            elif fit < self.delta_fitness:
                self.delta_fitness = fit
                self.delta_position = self.population[i].copy()

    def _update_positions(self, a):
        """
        Update the positions of the wolves in the population.
        """
        for i in range(self.population_size):
            wolf = self.population[i]

            # Calculate new positions based on alpha, beta, and delta wolves
            D_alpha = abs(np.random.uniform(0, 1) * self.alpha_position - wolf)
            D_beta = abs(np.random.uniform(0, 1) * self.beta_position - wolf)
            D_delta = abs(np.random.uniform(0, 1) * self.delta_position - wolf)

            A1 = 2 * a * np.random.uniform(0, 1) - a
            A2 = 2 * a * np.random.uniform(0, 1) - a
            A3 = 2 * a * np.random.uniform(0, 1) - a

            C1 = 2 * np.random.uniform(0, 1)
            C2 = 2 * np.random.uniform(0, 1)
            C3 = 2 * np.random.uniform(0, 1)

            X1 = self.alpha_position - A1 * D_alpha
            X2 = self.beta_position - A2 * D_beta
            X3 = self.delta_position - A3 * D_delta

            new_position = (X1 + X2 + X3) / 3

            # Ensure new position is within bounds
            self.population[i] = np.clip(new_position, self.variable_bounds[:, 0], self.variable_bounds[:, 1])

    def run(self):
        """
        Run the Grey Wolf Optimizer.

        Returns:
            tuple: Best solution and its objective function value.
        """
        for iteration in range(self.max_iterations):
            # Update alpha, beta, and delta wolves
            self._update_leadership()

            # Calculate the coefficient a, which decreases linearly over iterations
            a = 2 - iteration * (2 / self.max_iterations)

            # Update wolf positions
            self._update_positions(a)

            # (Optional) Uncomment to track progress
            # print(f"Iteration {iteration+1}, Alpha Fitness: {self.alpha_fitness}")

        return self.alpha_position, self.alpha_fitness


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    # Example: Minimize the Sphere function
    def sphere_function(x):
        return sum(xi ** 2 for xi in x)


    # Bounds for each variable
    variable_bounds = [(-10, 10)] * 5  # 5-dimensional Sphere function

    # Instantiate the GWO optimizer
    gwo = GreyWolfOptimizer(
        objective_function=sphere_function,
        variable_bounds=variable_bounds,
        population_size=30,
        max_iterations=100,
        seed=42
    )

    # Run the optimizer
    best_solution, best_fitness = gwo.run()
    print("Best Solution:", best_solution)
    print("Best Objective Value:", best_fitness)
