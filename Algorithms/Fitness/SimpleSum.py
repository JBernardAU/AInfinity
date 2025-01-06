import numpy as np

from Algorithms.Fitness.FitnessFunction import FitnessFunction


class SimpleSum(FitnessFunction):
    def __init__(self):
        """
        Initialize any parameters or state needed by the fitness function.
        """
        super().__init__()

    def compute_fitness(self, solution):
        """
        Compute a fitness score for the given gene sequence.

        :param solution: A NumPy array (or list) representing an individual's genes.
        :return: A float representing the fitness score.
        """
        # Example: sum of squares of the gene_sequence
        # You can replace this with whatever computation you like.
        solution = np.array(solution, dtype=float)  # Ensure we work with floats
        return float(np.sum(solution))

# Example usage:
if __name__ == "__main__":
    # Create an instance of the fitness function
    fitness = SimpleSum()

    # Suppose we have a gene sequence
    example = [1, -2, 3.5, 4]

    # Compute the fitness
    score = fitness.compute_fitness(example)

    print("Gene sequence:", example)
    print("Fitness score:", score)
