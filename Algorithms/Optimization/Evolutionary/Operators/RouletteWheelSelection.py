import numpy as np
import copy
from Algorithms.Optimization.Evolutionary.Operators.SelectionOperator import SelectionOperator

class RouletteSelection(SelectionOperator):
    def __init__(self):
        """
        Roulette wheel selection operator.

        """
        super().__init__()

    def select_parents(self, population, population_size):
        """
        Select parents using roulette wheel selection.

        Args:
            population (list): List of individuals in the population.
            population_size (int): Number of individuals in the population.
        Returns:
            list: Selected parents.
        """
        fitness_scores = [individual.fitness for individual in population]
        total_fitness = np.sum(fitness_scores)
        probabilities = fitness_scores / total_fitness
        parents_indices = np.random.choice(
            np.arange(population_size),
            size=population_size,
            p=probabilities
        )
        # Deep copy the selected parents
        selected_parents = [copy.deepcopy(population[i]) for i in parents_indices]
        return selected_parents