import numpy as np
import random


class GravitationalSearchAlgorithm:
    """
    Implementation of the Gravitational Search Algorithm (GSA) for optimization problems.
    """

    def __init__(self,
                 objective_function,
                 variable_bounds,
                 num_agents=30,
                 max_iterations=100,
                 G0=100,
                 alpha=20,
                 seed=None):
        """
        Initialize the Gravitational Search Algorithm.

        Parameters:
            objective_function (callable): The function to minimize.
            variable_bounds (list[tuple]): Bounds for each variable (lower, upper).
            num_agents (int): Number of agents (particles).
            max_iterations (int): Maximum number of iterations.
            G0 (float): Initial gravitational constant.
            alpha (float): Decay rate of the gravitational constant.
            seed (int, optional): Random seed for reproducibility.
        """
        self.objective_function = objective_function
        self.variable_bounds = np.array(variable_bounds)
        self.num_agents = num_agents
        self.max_iterations = max_iterations
        self.G0 = G0
        self.alpha = alpha
        self.num_variables = len(variable_bounds)
        self.seed = seed

        # Random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Initialize agent positions and velocities
        self.agents = np.random.uniform(
            self.variable_bounds[:, 0],
            self.variable_bounds[:, 1],
            (self.num_agents, self.num_variables)
        )
        self.velocities = np.zeros((self.num_agents, self.num_variables))

        # Initialize fitness and masses
        self.fitness = np.full(self.num_agents, np.inf)
        self.masses = np.zeros(self.num_agents)

        # Best solution found
        self.best_solution = None
        self.best_fitness = np.inf

    def _update_fitness_and_masses(self):
        """
        Evaluate the fitness of each agent and update their masses.
        """
        # Evaluate fitness for all agents
        self.fitness = np.array([self.objective_function(agent) for agent in self.agents])

        # Update the best solution found
        min_fitness = np.min(self.fitness)
        if min_fitness < self.best_fitness:
            self.best_fitness = min_fitness
            self.best_solution = self.agents[np.argmin(self.fitness)].copy()

        # Normalize fitness to calculate masses
        worst_fitness = np.max(self.fitness)
        if worst_fitness != min_fitness:  # Avoid division by zero
            normalized_fitness = (self.fitness - worst_fitness) / (min_fitness - worst_fitness)
            self.masses = normalized_fitness / np.sum(normalized_fitness)
        else:
            self.masses = np.ones(self.num_agents) / self.num_agents

    def _calculate_gravitational_force(self, iteration):
        """
        Calculate the gravitational forces acting on each agent.
        """
        G = self.G0 * np.exp(-self.alpha * iteration / self.max_iterations)  # Gravitational constant decay
        forces = np.zeros_like(self.agents)

        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    # Distance between agents
                    distance = np.linalg.norm(self.agents[i] - self.agents[j]) + 1e-10  # Avoid division by zero
                    # Gravitational force
                    force_magnitude = G * self.masses[i] * self.masses[j] / distance
                    # Directional vector
                    direction = (self.agents[j] - self.agents[i]) / distance
                    # Update total force
                    forces[i] += force_magnitude * direction
        return forces

    def _update_velocities_and_positions(self, forces):
        """
        Update the velocities and positions of the agents based on forces.
        """
        for i in range(self.num_agents):
            # Random weights for stochastic behavior
            r = np.random.uniform(size=self.num_variables)
            self.velocities[i] = r * self.velocities[i] + forces[i]
            self.agents[i] += self.velocities[i]

            # Ensure agents stay within bounds
            self.agents[i] = np.clip(self.agents[i], self.variable_bounds[:, 0], self.variable_bounds[:, 1])

    def run(self):
        """
        Run the Gravitational Search Algorithm.

        Returns:
            tuple: Best solution and its objective function value.
        """
        for iteration in range(self.max_iterations):
            # Step 1: Update fitness and masses
            self._update_fitness_and_masses()

            # Step 2: Calculate gravitational forces
            forces = self._calculate_gravitational_force(iteration)

            # Step 3: Update velocities and positions
            self._update_velocities_and_positions(forces)

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

    # Instantiate the GSA optimizer
    gsa = GravitationalSearchAlgorithm(
        objective_function=sphere_function,
        variable_bounds=variable_bounds,
        num_agents=30,
        max_iterations=100,
        G0=100,
        alpha=20,
        seed=42
    )

    # Run the optimizer
    best_solution, best_fitness = gsa.run()
    print("Best Solution:", best_solution)
    print("Best Objective Value:", best_fitness)
