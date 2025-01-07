from abc import ABC, abstractmethod

class Individual(ABC):
    """
    Abstract base class representing an individual in a genetic algorithm.
    Each individual has:
      - A name (string),
      - A list (or array) of values (genes),
      - A fitness value (numeric).
    """

    def __init__(self, name, values, fitness=0.0, fitness_function=None, mutator=None):
        """
        :param name: A string name/identifier for the individual.
        :param values: Initial value array.
        :param fitness: A numeric value representing the individual's fitness. Default is 0.
        """
        self._name = name
        self._values = values
        self._fitness = fitness
        self._fitness_function = fitness_function
        self._mutation_op = mutator
        self.mutate_self()

    @property
    def name(self):
        """Return the individual's name."""
        return self._name

    @name.setter
    def name(self, new_name):
        """Set the individual's name."""
        self._name = new_name

    @property
    def values(self):
        """Return the list/array of values (genes)."""
        return self._values

    @values.setter
    def values(self, new_values):
        """Set the list/array of values (genes)."""
        self._values = new_values

    @property
    def fitness(self):
        """Return the individual's fitness."""
        return self._fitness

    @fitness.setter
    def fitness(self, new_fitness):
        """Set the individual's fitness."""
        self._fitness = new_fitness

    @property
    def mutation_op(self):
        """Return the mutator object."""
        return self._mutation_op

    @mutation_op.setter
    def mutation_op(self, new_mutator):
        """Set a new mutator."""
        self._mutation_op = new_mutator

    @property
    def fitness_function(self):
        """Return the mutator object."""
        return self._fitness_function

    @fitness_function.setter
    def fitness_function(self, new_fitness_fn):
        """Set a new mutator."""
        self._fitness_function = new_fitness_fn

    def evaluate_fitness(self):
        """
        Example fitness calculation: sum of the individual's values.
        You can replace this logic with any domain-specific fitness evaluation.
        """
        if not self._values:
            self._fitness = 0.0
        else:
            self._fitness = self.fitness_function.compute_fitness(self._values)

    def mutate_self(self):
        """
        Use the mutator to randomly mutate the individual's values.
        This is just an example method that calls the mutator if one is available.
        """
        if self._mutation_op is not None:
            self._values = self._mutation_op.mutate_individual(self._values, forced=True)
