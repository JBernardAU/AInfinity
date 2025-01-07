from abc import ABC, abstractmethod


class MutationOperator(ABC):
    """
    Abstract parent class for mutation operators in a genetic algorithm.
    """

    def __init__(self, mutation_rate, min_values, max_values, gene_types):
        """
        Initialize common attributes for any mutation operator.

        :param mutation_rate: Probability of mutating a given gene (0 to 1).
        :param min_values: List/array of minimum values for each gene.
        :param max_values: List/array of maximum values for each gene.
        :param gene_types: List/array of types ('int' or 'float') for each gene.
        """
        self.mutation_rate = mutation_rate
        self.min_values = min_values
        self.max_values = max_values
        self.gene_types = gene_types

    @abstractmethod
    def mutate_individual(self, individual, forced=False):
        """
        Mutate a single individual. Must be overridden by subclasses.

        :param individual: A list/array representing the individual's genes.
        :param forced: Determined if mutation should always occur regardless of mutation rate.
        :return: A potentially mutated version of the individual.
        """
        pass

    @abstractmethod
    def mutate_population(self, population):
        """
        Mutate an entire population. Must be overridden by subclasses.

        :param population: A list of individuals, each one a list/array of genes.
        :return: A new (mutated) list of individuals of the same size.
        """
        pass
