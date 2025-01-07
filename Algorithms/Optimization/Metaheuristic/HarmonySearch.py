import random


class HarmonySearch:
    """
    A class implementing the Harmony Search algorithm for optimization problems.
    """

    def __init__(self,
                 objective_function,
                 variable_bounds,
                 harmony_memory_size=10,
                 harmony_memory_consideration_rate=0.9,
                 pitch_adjustment_rate=0.3,
                 bandwidth=0.01,
                 max_iterations=100,
                 seed=None):
        """
        Initialize the Harmony Search optimizer.

        Parameters:
            objective_function (callable): The function to minimize/maximize.
            variable_bounds (list[tuple]): Bounds for each variable (lower, upper).
            harmony_memory_size (int): Number of harmonies stored in memory.
            harmony_memory_consideration_rate (float): Probability of picking from memory.
            pitch_adjustment_rate (float): Probability of pitch adjustment.
            bandwidth (float): Range for pitch adjustment (scaled by variable range).
            max_iterations (int): Maximum number of iterations.
            seed (int, optional): Random seed for reproducibility.
        """
        self.objective_function = objective_function
        self.variable_bounds = variable_bounds
        self.harmony_memory_size = harmony_memory_size
        self.hmcr = harmony_memory_consideration_rate
        self.par = pitch_adjustment_rate
        self.bandwidth = bandwidth
        self.max_iterations = max_iterations
        self.num_variables = len(variable_bounds)

        # Random seed for reproducibility
        if seed is not None:
            random.seed(seed)

        # Initialize Harmony Memory (HM)
        self.harmony_memory = []
        self.harmony_memory_scores = []

    def _initialize_harmony_memory(self):
        """
        Initialize the Harmony Memory with random solutions within bounds.
        """
        self.harmony_memory = [
            [random.uniform(bounds[0], bounds[1]) for bounds in self.variable_bounds]
            for _ in range(self.harmony_memory_size)
        ]
        self.harmony_memory_scores = [
            self.objective_function(harmony) for harmony in self.harmony_memory
        ]

    def _improvise_new_harmony(self):
        """
        Improvise a new harmony by considering HMCR, PAR, and random generation.
        """
        new_harmony = []
        for i in range(self.num_variables):
            if random.random() < self.hmcr:  # Memory consideration
                # Choose a value from Harmony Memory
                value = random.choice([h[i] for h in self.harmony_memory])
                if random.random() < self.par:  # Pitch adjustment
                    value += self.bandwidth * (
                                random.uniform(-1, 1) * (self.variable_bounds[i][1] - self.variable_bounds[i][0]))
            else:  # Random consideration
                value = random.uniform(self.variable_bounds[i][0], self.variable_bounds[i][1])

            # Ensure the value is within bounds
            value = max(min(value, self.variable_bounds[i][1]), self.variable_bounds[i][0])
            new_harmony.append(value)

        return new_harmony

    def _update_harmony_memory(self, new_harmony):
        """
        Update the Harmony Memory by replacing the worst solution if the new solution is better.
        """
        new_score = self.objective_function(new_harmony)
        worst_score_idx = max(range(self.harmony_memory_size), key=lambda idx: self.harmony_memory_scores[idx])

        if new_score < self.harmony_memory_scores[worst_score_idx]:
            # Replace the worst harmony
            self.harmony_memory[worst_score_idx] = new_harmony
            self.harmony_memory_scores[worst_score_idx] = new_score

    def run(self):
        """
        Run the Harmony Search algorithm.

        Returns:
            tuple: Best solution and its objective function value.
        """
        # Step 1: Initialize Harmony Memory
        self._initialize_harmony_memory()

        # Step 2: Iterative improvisation and memory update
        for iteration in range(self.max_iterations):
            new_harmony = self._improvise_new_harmony()
            self._update_harmony_memory(new_harmony)

            # (Optional) Uncomment to track progress
            # print(f"Iteration {iteration+1}, Best Score: {min(self.harmony_memory_scores)}")

        # Return the best harmony found
        best_idx = min(range(self.harmony_memory_size), key=lambda idx: self.harmony_memory_scores[idx])
        return self.harmony_memory[best_idx], self.harmony_memory_scores[best_idx]


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    # Example: Minimize the Sphere function
    def sphere_function(x):
        return sum(xi ** 2 for xi in x)


    # Bounds for each variable
    variable_bounds = [(-5.12, 5.12)] * 5  # 5-dimensional Sphere function

    # Instantiate Harmony Search optimizer
    hs = HarmonySearch(objective_function=sphere_function,
                       variable_bounds=variable_bounds,
                       harmony_memory_size=10,
                       harmony_memory_consideration_rate=0.9,
                       pitch_adjustment_rate=0.3,
                       bandwidth=0.01,
                       max_iterations=100,
                       seed=42)

    # Run the optimizer
    best_solution, best_score = hs.run()
    print("Best Solution:", best_solution)
    print("Best Objective Value:", best_score)
