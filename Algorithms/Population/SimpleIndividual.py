from Algorithms.Population.Individual import Individual


class SimpleIndividual(Individual):
    """
    A concrete implementation of Individual, which also has
    a mutator to randomize (mutate) its values.
    """

    def __init__(self, name, values, fitness=0.0, fitness_function=None, mutator=None):
        """
        :param name: Name of the individual.
        :param values: Initial value array.
        :param fitness: Initial fitness value (default 0).
        :param mutator: An object with a mutate_individual(values) method.
        """
        super().__init__(name, values, fitness, fitness_function, mutator)

    def __repr__(self):
        return (f"Individual(name='{self._name}', "
                f"values={self._values}, fitness={self._fitness})")


# Example usage (assuming you already have a mutator with a mutate_individual method):
if __name__ == "__main__":
    min_vals = [0, 0.5, 1, 2.0]  # Minimum values for each gene
    max_vals = [10, 2.5, 10, 5.0]  # Maximum values for each gene
    gene_types = ["int", "float", "int", "float"]  # Mixed gene types

    from Algorithms.Fitness.SimpleSum import SimpleSum
    from Algorithms.Optimization.SimpleMutatation import SimpleMutation

    fitness_fn = SimpleSum()
    mutation_op = SimpleMutation(
        mutation_rate=0.01,
        min_values=min_vals,
        max_values=max_vals,
        value_types=gene_types
    )

    # Create an Individual with a name, values, and attach a mock mutator
    ind = SimpleIndividual(name="Alpha", values=min_vals, fitness_function=fitness_fn, mutator=mutation_op)

    # Evaluate fitness before mutation
    ind.evaluate_fitness()
    print("Before mutation:", ind)

    # Mutate this individual
    ind.mutate_self()
    ind.evaluate_fitness()
    print("After mutation:", ind)

    print(ind.values)
