import numpy as np


class CuckooSearch:
    """
    Implementation of the Cuckoo Search algorithm for optimization problems.
    """

    def __init__(self,
                 objective_function,
                 variable_bounds,
                 population_size=25,
                 max_iterations=100,
                 discovery_rate=0.25,
                 seed=None):
        """
        Initialize the Cuckoo Search optimizer.

        Parameters:
            objective_function (callable): The function to minimize.
            variable_bounds (list[tuple]): Bounds for each variable (lower, upper).
            population_size (int): Number of nests in the population.
            max_iterations (int): Maximum number of iterations.
            discovery_rate (float): Probability of discovering and replacing cuckoo eggs.
            seed (int, optional): Random seed for reproducibility.
        """
        self.objective_function = objective_function
        self.variable_bounds = np.array(variable_bounds)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.discovery_rate = discovery_rate
        self.num_variables = len(variable_bounds)
        self.seed = seed

        # Random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Initialize the population (nests) randomly within bounds
        self.nests = np.random.uniform(
            self.variable_bounds[:, 0],
            self.variable_bounds[:, 1],
            (self.population_size, self.num_variables)
        )
        self.fitness = np.array([np.inf] * self.population_size)

        # Best solution
        self.best_nest = None
        self.best_fitness = np.inf

    def _evaluate_fitness(self):
        """
        Evaluate the fitness of all nests and update the best solution.
        """
        for i in range(self.population_size):
            fit = self.objective_function(self.nests[i])
            if fit < self.fitness[i]:
                self.fitness[i] = fit
                if fit < self.best_fitness:
                    self.best_fitness = fit
                    self.best_nest = self.nests[i].copy()

    def _levy_flight(self, cuckoo):
        """
        Perform Levy flight to generate a new solution.
        """
        beta = 1.5  # Levy exponent
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=self.num_variables)
        v = np.random.normal(0, 1, size=self.num_variables)
        step = u / abs(v) ** (1 / beta)
        step_size = 0.01 * step * (self.nests[np.random.randint(self.population_size)] - cuckoo)
        new_solution = cuckoo + step_size
        return np.clip(new_solution, self.variable_bounds[:, 0], self.variable_bounds[:, 1])

    def _replace_worst_nests(self):
        """
        Replace a fraction of the worst nests with random solutions.
        """
        num_replace = int(self.discovery_rate * self.population_size)
        worst_indices = np.argsort(self.fitness)[-num_replace:]
        for i in worst_indices:
            self.nests[i] = np.random.uniform(
                self.variable_bounds[:, 0],
                self.variable_bounds[:, 1],
                self.num_variables
            )
            self.fitness[i] = self.objective_function(self.nests[i])

    def run(self):
        """
        Run the Cuckoo Search algorithm.

        Returns:
            tuple: Best solution and its objective function value.
        """
        for iteration in range(self.max_iterations):
            # Step 1: Generate a new cuckoo and evaluate it
            for i in range(self.population_size):
                cuckoo = self.nests[i]
                new_cuckoo = self._levy_flight(cuckoo)
                new_fitness = self.objective_function(new_cuckoo)

                # Replace if the new solution is better
                if new_fitness < self.fitness[i]:
                    self.nests[i] = new_cuckoo
                    self.fitness[i] = new_fitness

                    # Update the global best if necessary
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_nest = new_cuckoo.copy()

            # Step 2: Replace a fraction of the worst nests
            self._replace_worst_nests()

            # Evaluate fitness after replacement
            self._evaluate_fitness()

            # (Optional) Uncomment to track progress
            # print(f"Iteration {iteration+1}, Best Fitness: {self.best_fitness}")

        return self.best_nest, self.best_fitness


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    # Example: Minimize the Sphere function
    def sphere_function(x):
        return sum(xi ** 2 for xi in x)


    # Bounds for each variable
    variable_bounds = [(-10, 10)] * 5  # 5-dimensional Sphere function

    # Instantiate the Cuckoo Search optimizer
    cs = CuckooSearch(
        objective_function=sphere_function,
        variable_bounds=variable_bounds,
        population_size=25,
        max_iterations=100,
        discovery_rate=0.25,
        seed=42
    )

    # Run the optimizer
    best_solution, best_fitness = cs.run()
    print("Best Solution:", best_solution)
    print("Best Objective Value:", best_fitness)
