import numpy as np
import matplotlib.pyplot as plt


def create_histogram(array, bin_size, x_range=None):
    """
    Create a histogram from an N-dimensional array.

    Parameters:
    - array (np.ndarray): Input N-dimensional array with values ranging from 0 to x.
    - bin_size (int): Size of each histogram bucket.
    - x_range (tuple, optional): Tuple specifying the range (min, max) for the histogram.
                                 If None, it defaults to the array's min and max values.

    Returns:
    - bin_edges (np.ndarray): Array of bin edges.
    - histogram (np.ndarray): Histogram counts for each bin.
    """
    # Flatten the array to 1D
    flat_array = array.flatten()

    # Determine the range for the histogram
    if x_range is None:
        x_range = (flat_array.min(), flat_array.max())

    # Generate bins based on bin size
    bins = np.arange(x_range[0], x_range[1] + bin_size, bin_size)

    # Compute the histogram
    histogram, bin_edges = np.histogram(flat_array, bins=bins)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(flat_array, bins=bins, edgecolor='k', alpha=0.7)
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    return bin_edges, histogram

# Example Usage:
# Uncomment below lines to test with a sample N-dimensional array
#sample_array = np.random.randint(0, 100, size=(4, 5, 6))  # Example 3D array
#create_histogram(sample_array, bin_size=1)
