import numpy as np
import copy
from Algorithms.Optimization.Evolutionary.Operators.SelectionOperator import SelectionOperator

class TournamentSelection(SelectionOperator):
    def __init__(self, tournament_size):
        """
        Tournament selection operator.

        Args:
            tournament_size (int): Number of individuals in each tournament.
        """
        super().__init__()
        self.tournament_size = tournament_size

    def select_parents(self, population, population_size):
        """
        Select parents using tournament selection.

        Args:
            population (list): List of individuals in the population.
            population_size (int): Number of individuals in the population.
        Returns:
            list: Selected parents.
        """
        selected_parents = []
        for _ in range(population_size):
            # Randomly select individuals for the tournament
            tournament_indices = np.random.choice(
                np.arange(population_size),
                size=self.tournament_size,
                replace=False
            )
            tournament = [population[i] for i in tournament_indices]
            # Select the individual with the highest fitness
            best_individual = max(tournament, key=lambda ind: ind.fitness)
            selected_parents.append(copy.deepcopy(best_individual))
        return selected_parents

# Example usage
if __name__ == "__main__":
    class Individual:
        def __init__(self, fitness):
            self.fitness = fitness

    # Create a mock population
    population = [Individual(fitness=np.random.randint(1, 100)) for _ in range(10)]
    population_size = len(population)
    tournament_size = 3

    # Initialize tournament selection operator
    selector = TournamentSelection(tournament_size)

    # Select parents
    parents = selector.select_parents(population, population_size)
    print("Selected parents' fitness values:", [parent.fitness for parent in parents])
