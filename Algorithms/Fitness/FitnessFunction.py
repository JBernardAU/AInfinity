from abc import ABC, abstractmethod

class FitnessFunction(ABC):
    def __init__(self):
        """
        Initialize any parameters or state needed by the fitness function.
        """
        pass

    @abstractmethod
    def compute_fitness(self, gene_sequence):
        """
        Compute a fitness score for the given gene sequence.

        :param gene_sequence: A NumPy array (or list) representing an individual's genes.
        :return: A float representing the fitness score.
        """
        pass

if __name__ == "__main__":
    example = [1, -2, 3.5, 4]
    fitness = FitnessFunction()
    print(fitness.compute_fitness(example))
