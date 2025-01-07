import random
import numpy as np


class Crossover:
    def __init__(self, crossover_rate, gene_length):
        """
        Initialize the SinglePointCrossover instance.

        :param crossover_rate: Probability of performing a crossover (float between 0 and 1).
        :param gene_length: Length of the gene sequences (integer).
        """
        self.crossover_rate = crossover_rate
        self.gene_length = gene_length

    def crossover(self, parent1, parent2):
        pass

class SinglePointCrossover(Crossover):
    def __init__(self, crossover_rate, gene_length):
        """
        Initialize the SinglePointCrossover instance.

        :param crossover_rate: Probability of performing a crossover (float between 0 and 1).
        :param gene_length: Length of the gene sequences (integer).
        """
        super().__init__(crossover_rate, gene_length)

    def crossover(self, parent1, parent2):
        """
        Perform single-point crossover between two parents.

        :param parent1: First parent sequence (numpy array).
        :param parent2: Second parent sequence (numpy array).
        :return: A tuple containing two offspring (numpy arrays).
        """
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.gene_length - 1)
            child1 = np.concatenate((parent1.values[:point], parent2.values[point:]))
            child2 = np.concatenate((parent2.values[:point], parent1.values[point:]))
            return child1, child2
        return parent1, parent2

# Example usage:
# crossover_instance = SinglePointCrossover(crossover_rate=0.8, gene_length=10)
# offspring1, offspring2 = crossover_instance.crossover(parent1, parent2)