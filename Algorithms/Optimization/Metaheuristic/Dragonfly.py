import numpy as np


class DragonflyAlgorithm:
    """
    Implementation of the Dragonfly Algorithm (DA) for optimization problems.
    """

    def __init__(self,
                 objective_function,
                 variable_bounds,
                 population_size=30,
                 max_iterations=100,
                 weights=(0.1, 0.1, 0.1, 1.0, 1.0),
                 inertia=0.9,
                 seed=None):
        """
        Initialize the Dragonfly Algorithm.

        Parameters:
            objective_function (callable): The function to minimize.
            variable_bounds (list[tuple]): Bounds for each variable (lower, upper).
            population_size (int): Number of dragonflies in the swarm.
            max_iterations (int): Maximum number of iterations.
            weights (tuple): Weights for separation, alignment, cohesion, attraction, and distraction.
            inertia (float): Inertia weight for velocity update.
            seed (int, optional): Random seed for reproducibility.
        """
        self.objective_function = objective_function
        self.variable_bounds = np.array(variable_bounds)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.weights = weights  # Separation, alignment, cohesion, attraction, distraction
        self.inertia = inertia
        self.num_variables = len(variable_bounds)
        self.seed = seed

        # Random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Initialize positions and velocities randomly
        self.positions = np.random.uniform(
            self.variable_bounds[:, 0],
            self.variable_bounds[:, 1],
            (self.population_size, self.num_variables)
        )
        self.velocities = np.zeros((self.population_size, self.num_variables))

        # Initialize best solutions
        self.fitness = np.full(self.population_size, np.inf)
        self.best_global = None
        self.best_global_fitness = np.inf

    def _evaluate_fitness(self):
        """
        Evaluate the fitness of all dragonflies and update the global best.
        """
        for i in range(self.population_size):
            fit = self.objective_function(self.positions[i])
            if fit < self.fitness[i]:
                self.fitness[i] = fit
                if fit < self.best_global_fitness:
                    self.best_global_fitness = fit
                    self.best_global = self.positions[i].copy()

    def _calculate_social_factors(self):
        """
        Calculate the social interaction components: separation, alignment, and cohesion.
        """
        separation = np.zeros((self.population_size, self.num_variables))
        alignment = np.zeros((self.population_size, self.num_variables))
        cohesion = np.zeros((self.population_size, self.num_variables))

        for i in range(self.population_size):
            neighbors = [
                j for j in range(self.population_size)
                if np.linalg.norm(self.positions[i] - self.positions[j]) < 1.0 and i != j
            ]
            if neighbors:
                neighbor_positions = self.positions[neighbors]
                neighbor_velocities = self.velocities[neighbors]

                # Separation: Avoid collisions
                separation[i] = -np.sum(self.positions[i] - neighbor_positions, axis=0)

                # Alignment: Match the average velocity of neighbors
                alignment[i] = np.mean(neighbor_velocities, axis=0)

                # Cohesion: Move towards the center of neighbors
                cohesion[i] = np.mean(neighbor_positions, axis=0) - self.positions[i]

        return separation, alignment, cohesion

    def _update_positions_and_velocities(self, separation, alignment, cohesion, iteration):
        """
        Update the positions and velocities of the dragonflies.
        """
        w = self.inertia * (1 - iteration / self.max_iterations)  # Inertia weight
        s_weight, a_weight, c_weight, f_weight, e_weight = self.weights

        for i in range(self.population_size):
            # Attraction to food (best global position)
            attraction = self.best_global - self.positions[i]

            # Distraction by enemies (repulsion from the worst position)
            distraction = np.random.uniform(-1, 1, self.num_variables)  # Simplified as random noise

            # Velocity update
            self.velocities[i] = (
                    w * self.velocities[i]
                    + s_weight * separation[i]
                    + a_weight * alignment[i]
                    + c_weight * cohesion[i]
                    + f_weight * attraction
                    - e_weight * distraction
            )

            # Update position
            self.positions[i] += self.velocities[i]

            # Ensure positions are within bounds
            self.positions[i] = np.clip(self.positions[i], self.variable_bounds[:, 0], self.variable_bounds[:, 1])

    def run(self):
        """
        Run the Dragonfly Algorithm.

        Returns:
            tuple: Best solution and its objective function value.
        """
        for iteration in range(self.max_iterations):
            # Step 1: Evaluate fitness
            self._evaluate_fitness()

            # Step 2: Calculate social factors
            separation, alignment, cohesion = self._calculate_social_factors()

            # Step 3: Update positions and velocities
            self._update_positions_and_velocities(separation, alignment, cohesion, iteration)

            # (Optional) Uncomment to track progress
            # print(f"Iteration {iteration+1}, Best Fitness: {self.best_global_fitness}")

        return self.best_global, self.best_global_fitness


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    # Example: Minimize the Sphere function
    def sphere_function(x):
        return sum(xi ** 2 for xi in x)


    # Bounds for each variable
    variable_bounds = [(-10, 10)] * 5  # 5-dimensional Sphere function

    # Instantiate the Dragonfly Algorithm optimizer
    da = DragonflyAlgorithm(
        objective_function=sphere_function,
        variable_bounds=variable_bounds,
        population_size=30,
        max_iterations=100,
        weights=(0.1, 0.1, 0.1, 1.0, 1.0),
        inertia=0.9,
        seed=42
    )

    # Run the optimizer
    best_solution, best_fitness = da.run()
    print("Best Solution:", best_solution)
    print("Best Objective Value:", best_fitness)
