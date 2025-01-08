import numpy as np
import importlib
import copy

from Algorithms.Optimization.Evolutionary.BoundsConfiguration import BoundsConfiguration

class GeneticAlgorithm:
    def __init__(self, population_size, gene_configuration, fitness_function,
                 selection_op, crossover_op, mutation_op, survival_op, individual_type, generations=100):
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
        self.gene_config = gene_configuration
        self.fitness_function = fitness_function
        self.selection_op = selection_op
        self.crossover_op = crossover_op
        self.mutation_op = mutation_op
        self.generations = generations
        self.survival_op = survival_op
        self.individual_class = getattr(importlib.import_module("Algorithms.Population." + individual_type), individual_type)
        if self.individual_class is None:
            raise ImportError("The module does not define a class named '" + individual_type + "'.")

        self.best_individual = None
        self.population = []
        self.initialize_population()

        """
        These properties are used for internal monitoring and are not required for the algorithm to function
        parent1 and parent2 are Individual objects
        child1 and parent2 are Individual objects
        """
        self.parent1 = None
        self.parent2 = None
        self.child1 = None
        self.child2 = None

    def initialize_population(self):
        """
        Randomly initialize the population using the gene configuration.

        :return: A numpy array of shape (population_size, gene_length) containing the initialized population.
        """
        self.pre_population_initialization()
        # Create an instance of the dynamically loaded class
        self.population = np.array([self.individual_class(name="Alpha", values=self.gene_config.min_values,
                                                          fitness_function=self.fitness_function, mutator=self.mutation_op) for _ in
                                    range(self.population_size)])
        self.post_population_initialization()

    def evaluate_fitness(self):
        """
        Evaluate the fitness of each individual in the population.
        """
        self.pre_evaluate_fitness()
        for ind in self.population:
            self.pre_evaluate_fitness_individual()
            ind.fitness = self.fitness_function.compute_fitness(ind.values)
            self.post_evaluate_fitness_individual()
        self.best_individual = max(self.population, key=lambda individual: individual.fitness)
        self.post_evaluate_fitness()

    def select_parents(self):
        """
        Select parents using roulette wheel selection.
        """
        selected_parents = self.selection_op.select_parents(self.population, self.population_size)
        return selected_parents

    def crossover(self, parent1, parent2):
        """
        Perform single-point crossover between two parents.
        """
        self.pre_crossover()
        parent1, parent2 = self.crossover_op(parent1, parent2)
        self.post_crossover()
        return parent1, parent2

    def mutate(self, values):
        self.pre_mutate()
        values = self.mutation_op.mutate(values)
        self.post_mutate()
        return values

    def evolve(self):
        """
        Evolve the population over the specified number of generations.
        """
        for generation in range(self.generations):
            self.start_generation()
            self.evaluate_fitness()
            parents = self.select_parents()
            for i in range(0, self.population_size, 2):
                self.pre_parent_processing()
                self.parent1, self.parent2 = parents[i], parents[min(i + 1, self.population_size - 1)]
                # make a deep copy of the parents as the children
                # the parents are retained so the children can be compared to their parents
                self.child1 = copy.deepcopy(self.parent1)
                self.child2 = copy.deepcopy(self.parent2)
                self.child1.values, self.child2.values = self.crossover_op.crossover(self.child1.values, self.child2.values)
                self.mutate(self.child1.values)
                self.mutate(self.child1.values)
                self.population = np.append(self.population, self.child1)
                self.population = np.append(self.population, self.child2)
                self.post_parent_processing()
            self.population = np.array(self.survival_op.apply_survival(self.population, self.population_size))
            self.end_generation()
            print(f"Generation {generation + 1}: Best Fitness = {self.best_individual.fitness}")
        return self.best_individual

    def pre_population_initialization(self):
        pass

    def post_population_initialization(self):
        pass

    def pre_evaluate_fitness(self):
        pass

    def post_evaluate_fitness(self):
        pass

    def pre_evaluate_fitness_individual(self):
        pass

    def post_evaluate_fitness_individual(self):
        pass

    def pre_mutate(self):
        pass

    def post_mutate(self):
        pass

    def pre_crossover(self):
        pass

    def post_crossover(self):
        pass

    def pre_parent_processing(self):
        pass

    def post_parent_processing(self):
        pass

    def start_generation(self):
        pass

    def end_generation(self):
        pass


# Example usage:
if __name__ == "__main__":
    min_vals = [0, 0.5, 1, 2.0]  # Minimum values for each gene
    max_vals = [10, 2.5, 10, 5.0]  # Maximum values for each gene
    gene_types = ["int", "float", "int", "float"]  # Mixed gene types

    from Algorithms.Optimization.Evolutionary.Operators.RouletteWheelSelection import RouletteSelection
    from Algorithms.Optimization.Evolutionary.Operators.Crossover import SinglePointCrossover
    from Algorithms.Optimization.Evolutionary.Operators.SimpleMutatation import SimpleMutation
    from Algorithms.Optimization.Evolutionary.Operators.EliteSurvival import EliteSurvival
    from Algorithms.Fitness.SimpleSum import SimpleSum

    # Instantiate a GeneConfiguration with mixed type constraints
    simple_sum = SimpleSum()
    gene_config = BoundsConfiguration(min_vals, max_vals, gene_types)
    roulette_wheel = RouletteSelection()
    single_point_crossover =  SinglePointCrossover(0.8, len(gene_config))
    elite_survival = EliteSurvival(elitism_count=2)

    # Create a MutationOperator instance
    simple_mutation = SimpleMutation(
        mutation_rate=0.01,
        min_values=min_vals,
        max_values=max_vals,
        value_types=gene_types
    )

    # Initialize the genetic algorithm
    ga = GeneticAlgorithm(
        population_size=20,
        gene_configuration=gene_config,
        fitness_function=simple_sum,
        selection_op=roulette_wheel,
        crossover_op=single_point_crossover,
        mutation_op=simple_mutation,
        survival_op=elite_survival,
        individual_type="SimpleIndividual",
        generations=50
    )

    # Run the algorithm
    best_solution = ga.evolve()
    print("Best Solution Found:", best_solution)
    print("Fitness of Best Solution:", ga.fitness_function.compute_fitness(best_solution.values))