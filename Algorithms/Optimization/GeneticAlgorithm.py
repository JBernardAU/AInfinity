import numpy as np
from Algorithms.Optimization.GeneConfiguration import GeneConfiguration

class GeneticAlgorithm:
    def __init__(self, population_size, gene_config, fitness_function,
                 crossover, mutator, generations=100):
        """
        Initialize the Genetic Algorithm.

        Parameters:
        - population_size (int): Number of individuals in the population.
        - gene_length (int): Number of genes in each individual.
        - fitness_function (callable): Function to evaluate the fitness of an individual.
        - crossover_rate (float): Probability of crossover (default: 0.8).
        - mutation_rate (float): Probability of mutation (default: 0.01).
        - generations (int): Number of generations to evolve (default: 100).
        """
        self.population_size = population_size
        self.gene_config = gene_config
        self.fitness_function = fitness_function
        self.crossover = crossover
        self.mutator = mutator
        self.generations = generations
        self.population = self.initialize_population()
        self.best_individual = None

    def initialize_population(self):
        """
        Randomly initialize the population using the gene configuration.

        :return: A numpy array of shape (population_size, gene_length) containing the initialized population.
        """
        return np.array([self.gene_config.generate_gene() for _ in range(self.population_size)])

    def evaluate_fitness(self):
        """
        Evaluate the fitness of each individual in the population.
        """
        fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])
        self.best_individual = self.population[np.argmax(fitness_scores)]
        return fitness_scores

    def select_parents(self, fitness_scores):
        """
        Select parents using roulette wheel selection.
        """
        total_fitness = np.sum(fitness_scores)
        probabilities = fitness_scores / total_fitness
        parents_indices = np.random.choice(
            np.arange(self.population_size),
            size=self.population_size,
            p=probabilities
        )
        return self.population[parents_indices]

    def crossover(self, parent1, parent2):
        """
        Perform single-point crossover between two parents.
        """
        parent1, parent2 = self.crossover(parent1, parent2)
        return parent1, parent2

    def mutate(self, individual):
        individual = self.mutator.mutate_individual(individual)
        return individual

    def evolve(self):
        """
        Evolve the population over the specified number of generations.
        """
        for generation in range(self.generations):
            fitness_scores = self.evaluate_fitness()
            new_population = []
            parents = self.select_parents(fitness_scores)
            for i in range(0, self.population_size, 2):
                parent1, parent2 = parents[i], parents[min(i + 1, self.population_size - 1)]
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))
            self.population = np.array(new_population[:self.population_size])
            print(f"Generation {generation + 1}: Best Fitness = {np.max(fitness_scores)}")
        return self.best_individual

# Example usage:
if __name__ == "__main__":
    print("Sample Genetic Algorithm")
    min_vals = [0, 0.5, 1, 2.0]  # Minimum values for each gene
    max_vals = [10, 2.5, 10, 5.0]  # Maximum values for each gene
    gene_types = ["int", "float", "int", "float"]  # Mixed gene types

    from Algorithms.Optimization.SimpleMutatation import SimpleMutation
    from Algorithms.Optimization.Crossover import SinglePointCrossover
    from Algorithms.Fitness.SimpleSum import SimpleSum

    # Instantiate a GeneConfiguration with mixed type constraints
    gene_config = GeneConfiguration(min_vals, max_vals, gene_types)
    crossover =  SinglePointCrossover(0.8, len(gene_config))
    fitness_fn = SimpleSum()

    # Create a MutationOperator instance
    mutation_op = SimpleMutation(
        mutation_rate=0.01,
        min_values=min_vals,
        max_values=max_vals,
        value_types=gene_types
    )

    # Initialize the genetic algorithm
    ga = GeneticAlgorithm(
        population_size=20,
        gene_config=gene_config,
        fitness_function=fitness_fn,
        crossover=crossover,
        mutator=mutation_op,
        generations=50
    )

    # Run the algorithm
    best_solution = ga.evolve()
    print("Best Solution Found:", best_solution)
    print("Fitness of Best Solution:", fitness_function(best_solution))
