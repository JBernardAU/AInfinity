import numpy as np


class GeneConfiguration:
    def __init__(self, min_values, max_values, gene_types):
        """
        Initialize the GeneConfiguration instance with arrays of min and max values,
        and a corresponding array of types.

        :param min_values: Array-like of minimum values for each gene.
        :param max_values: Array-like of maximum values for each gene.
        :param gene_types: Array-like of types (e.g., 'int' or 'float') for each gene.
        """
        # Convert everything to NumPy arrays for easy indexing and validation
        self._min_values = np.array(min_values)
        self._max_values = np.array(max_values)
        self._gene_types = np.array(gene_types, dtype=object)  # dtype=object to allow string storage

        # Validate that all arrays have the same shape
        if (self.min_values.shape != self._max_values.shape or
                self.min_values.shape != self._gene_types.shape):
            raise ValueError("min_values, max_values, and gene_types must have the same shape.")

    def get_min_value(self, index):
        return self._min_values[index]

    def set_min_value(self, index, value):
        self._min_values[index] = value

    def get_max_value(self, index):
        return self._max_values[index]

    def set_max_value(self, index, value):
        self._max_values[index] = value

    @property
    def min_values(self):
        """Return the min values array."""
        return self._min_values

    @min_values.setter
    def min_values(self, new_min_values):
        """Set the min values array."""
        self._min_values = new_min_values

    @property
    def max_values(self):
        """Return the max values array."""
        return self._max_values

    @max_values.setter
    def max_values(self, new_max_values):
        """Set the max values array ."""
        self._max_values = new_max_values

    def __len__(self):
        return len(self.min_values)

    def generate_gene(self):
        """
        Generate a random gene sequence. The type of each gene (int or float) is determined
        by the corresponding element in self.gene_types.

        :return: A NumPy array representing the random gene sequence with mixed types.
        """
        gene_sequence = []
        for min_val, max_val, gene_type in zip(self.min_values,
                                               self._max_values,
                                               self._gene_types):

            if gene_type == "int":
                # np.random.randint generates an integer in [low, high),
                # so use max_val + 1 to make it inclusive if desired
                value = np.random.randint(min_val, max_val + 1)
            elif gene_type == "float":
                # np.random.uniform generates a float in [low, high)
                value = np.random.uniform(min_val, max_val)
            else:
                raise ValueError(f"Unsupported gene type: {gene_type}")

            gene_sequence.append(value)

        # Convert to a NumPy array; depending on mixing of ints/floats,
        # NumPy may upcast to float, or you can keep it as an object array.
        return np.array(gene_sequence, dtype=object)


# Example usage:
if __name__ == "__main__":
    min_vals = [0, 0.5, 1, 2.0]  # Minimum values for each gene
    max_vals = [10, 2.5, 10, 5.0]  # Maximum values for each gene
    gene_types = ["int", "float", "int", "float"]  # Mixed gene types

    # Instantiate a GeneConfiguration with mixed type constraints
    gene_config = GeneConfiguration(min_vals, max_vals, gene_types)

    # Generate a gene sequence
    gene_sequence = gene_config.generate_gene()
    print("Generated gene sequence:", gene_sequence)
    print("Types of each element:", [type(x) for x in gene_sequence])
