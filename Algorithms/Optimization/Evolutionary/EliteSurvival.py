import random
from Algorithms.Optimization.Evolutionary.SurvivalStrategy import SurvivalStrategy

class EliteSurvival(SurvivalStrategy):
    """
    A concrete class that preserves the top N individuals (elites) from the population
    based on their fitness property, then fills the remainder of the population.
    """

    def __init__(self, elitism_count):
        """
        :param elitism_count: Number of top individuals to preserve.
        """
        self.elitism_count = elitism_count

    def apply_survival(self, population, N):
        """
        Preserve the top N individuals from the population (by fitness) and return a
        newly formed population of the same size.

        :param population: A list of objects with 'name' and 'fitness' attributes.
        :param N: the size of the population
        :return: A new list (population) with the top N preserved.
        """
        population_size = N

        # Sort individuals by fitness in descending order
        sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)

        # Extract the top N elites
        elites = sorted_population[:self.elitism_count]

        # Start the new population with the elites
        new_population = list(elites)

        # Fill the rest of the new population by randomly sampling from the old population
        # but without allowing duplicates
        while len(new_population) < population_size:
            candidate = random.choice(population)
            if candidate not in new_population:
                new_population.append(candidate)

        return new_population


# Example model class for Individuals
class Individual:
    """
    Example class representing an individual with a name and a fitness property.
    """
    def __init__(self, name, fitness):
        self.name = name
        self.fitness = fitness

    def __repr__(self):
        return f"Individual(name='{self.name}', fitness={self.fitness})"


# Example usage:
if __name__ == "__main__":
    # Create a sample population
    population = [
        Individual("Individual_A", 10),
        Individual("Individual_B", 7),
        Individual("Individual_C", 15),
        Individual("Individual_D", 3),
        Individual("Individual_E", 12),
        Individual("Individual_F", 8),
    ]

    # Create an EliteSurvival strategy that preserves the top 2 individuals
    survival_strategy = EliteSurvival(elitism_count=2)

    # Apply survival
    new_pop = survival_strategy.apply_survival(population, 4)

    print("Old Population:", population)
    print("New Population:", new_pop)