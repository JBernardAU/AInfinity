from abc import ABC, abstractmethod

class SurvivalStrategy(ABC):
    """
    Abstract base class for survival strategies in a genetic algorithm.
    """

    @abstractmethod
    def apply_survival(self, population, N):
        """
        Preserve or select individuals from the population based on their
        fitness scores.

        :param population: A list of individuals (e.g., list of gene sequences).
        :param N: the size of the population
        :return: A new population (list) after applying the survival strategy.
        """
        pass

# Example usage
if __name__ == "__main__":
    # Suppose we have a population of 6 individuals
    population = [
        "Individual_A",
        "Individual_B",
        "Individual_C",
        "Individual_D",
        "Individual_E",
        "Individual_F"
    ]

    # Corresponding fitness scores (the larger, the better)
    fitness_scores = [10, 7, 15, 3, 12, 8]

    # Create an EliteSurvival instance to preserve the top 2 individuals
    survival_strategy = SurvivalStrategy()

    # Apply elite survival
    new_pop = survival_strategy.apply_survival(population, fitness_scores)

    print("Old Population:", population)
    print("Fitness Scores:", fitness_scores)
    print("New Population:", new_pop)
