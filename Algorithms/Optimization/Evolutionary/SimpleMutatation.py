import random
from Algorithms.Optimization.Evolutionary.MutationOperator import MutationOperator

class SimpleMutation(MutationOperator):
    """
    Mutation operator that can mutate gene values to a new random integer or float
    based on an array of types.
    """
    def __init__(self, mutation_rate, min_values, max_values, value_types):
        """
        :param mutation_rate: Probability of mutating a given gene (0 to 1).
        :param min_values: List/array of minimum values for each gene.
        :param max_values: List/array of maximum values for each gene.
        :param value_types: List/array of types ('int' or 'float') for each gene.
        """
        self.mutation_rate = mutation_rate
        self.min_values = min_values
        self.max_values = max_values
        self.gene_types = value_types

        # Optional validation: ensure all arrays match in length
        if not (len(self.min_values) == len(self.max_values) == len(self.gene_types)):
            raise ValueError("min_values, max_values, and gene_types must be of the same length.")

    def mutate_individual(self, individual, forced=False):
        """
        Mutate a single individual. Each gene in the individual has a probability
        of mutation_rate to be replaced with a new random value (int or float).

        :param forced:
        :param individual: List/array representing the individual's genes.
        :return: A new, potentially mutated, list of genes.
        """
        if len(individual) != len(self.gene_types):
            raise ValueError("Length of individual genes must match gene_types.")

        new_individual = []
        for i, current_value in enumerate(individual):
            if random.random() < self.mutation_rate or forced:
                # Mutate this gene
                mutated_value = self._choose_random(
                    self.min_values[i],
                    self.max_values[i],
                    self.gene_types[i]
                )
                new_individual.append(mutated_value)
            else:
                # Keep the current gene
                new_individual.append(current_value)

        return new_individual

    def mutate_population(self, population, forced=False):
        """
        Mutate an entire population of individuals.

        :param forced: Determined if mutation should always occur regardless of mutation rate
        :param population: List of individuals. Each individual is a list of genes.
        :return: A new population (list of mutated or unmutated individuals).
        """
        return [self.mutate_individual(ind, forced) for ind in population]

    def _choose_random(self, min_val, max_val, gene_type):
        """
        Internal helper to pick a random value within [min_val, max_val].
        Chooses int or float based on gene_type in a single function call.

        :param min_val: Minimum possible value for this gene.
        :param max_val: Maximum possible value for this gene.
        :param gene_type: 'int' or 'float'.
        :return: A random integer or float.
        """
        if gene_type == 'int':
            # randint is inclusive of max_val
            return random.randint(int(min_val), int(max_val))
        elif gene_type == 'float':
            return random.uniform(float(min_val), float(max_val))
        else:
            raise ValueError("gene_type must be 'int' or 'float'.")


# Example Usage
if __name__ == "__main__":
    # Suppose each individual has 3 genes: [gene0, gene1, gene2]
    # gene0: int between 0 and 5
    # gene1: float between -1.0 and 1.0
    # gene2: int between 10 and 15
    min_vals = [0,  -1.0, 10]
    max_vals = [5,   1.0, 15]
    gene_types = ['int', 'float', 'int']

    # Create a MutationOperator instance
    mutation_op = SimpleMutation(
        mutation_rate=0.5,  # 50% chance each gene mutates
        min_values=min_vals,
        max_values=max_vals,
        value_types=gene_types
    )

    # Example population of size 4 (each individual has 3 genes)
    population = [
        [2, 0.5, 12],
        [3, -0.3, 14],
        [5, 1.0, 10],
        [0, 0.0, 15],
    ]

    # Mutate the population
    mutated_population = mutation_op.mutate_population(population)

    print("Original population:")
    for ind in population:
        print(ind)

    print("\nMutated population:")
    for ind in mutated_population:
        print(ind)
