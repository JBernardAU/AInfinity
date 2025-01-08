from abc import ABC, abstractmethod
import numpy as np
import copy

class SelectionOperator(ABC):
    def __init__(self):
        """
        Abstract base class for selection operators.

        Args:
        """

    @abstractmethod
    def select_parents(self, population, population_size):
        """
        Abstract method for selecting parents.

        Returns:
            list: Selected parents.
        """
        pass