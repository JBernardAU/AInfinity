import numpy as np

class AntColonySystem:
    def __init__(self, num_ants, num_nodes, alpha, beta, evaporation_rate, pheromone_deposit, iterations, distance_matrix):
        self.num_ants = num_ants
        self.num_nodes = num_nodes
        self.alpha = alpha  # Pheromone importance
        self.beta = beta  # Distance importance
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.iterations = iterations
        self.distance_matrix = distance_matrix
        self.pheromone_matrix = np.ones((num_nodes, num_nodes)) / num_nodes

    def _initialize_ants(self):
        return [np.random.choice(self.num_nodes) for _ in range(self.num_ants)]

    def _update_pheromones(self, all_paths):
        self.pheromone_matrix *= (1 - self.evaporation_rate)  # Evaporate pheromones

        for path, length in all_paths:
            for i in range(len(path) - 1):
                from_node = path[i]
                to_node = path[i + 1]
                self.pheromone_matrix[from_node][to_node] += self.pheromone_deposit / length

    def _select_next_node(self, current_node, visited):
        probabilities = []
        for next_node in range(self.num_nodes):
            if next_node not in visited:
                pheromone = self.pheromone_matrix[current_node][next_node] ** self.alpha
                visibility = (1 / self.distance_matrix[current_node][next_node]) ** self.beta
                probabilities.append(pheromone * visibility)
            else:
                probabilities.append(0)

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        return np.random.choice(range(self.num_nodes), p=probabilities)

    def _construct_solution(self, ant):
        current_node = ant
        path = [current_node]
        visited = set(path)
        total_length = 0

        while len(visited) < self.num_nodes:
            next_node = self._select_next_node(current_node, visited)
            total_length += self.distance_matrix[current_node][next_node]
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node

        total_length += self.distance_matrix[current_node][path[0]]  # Return to start
        path.append(path[0])
        return path, total_length

    def optimize(self):
        best_path = None
        best_length = float('inf')

        for iteration in range(self.iterations):
            all_paths = []

            for ant in self._initialize_ants():
                path, length = self._construct_solution(ant)
                all_paths.append((path, length))

                if length < best_length:
                    best_path = path
                    best_length = length

            self._update_pheromones(all_paths)
            print(f"Iteration {iteration + 1}/{self.iterations}, Best Length: {best_length}")

        return best_path, best_length

# Example usage
def create_distance_matrix(num_nodes):
    np.random.seed(0)  # For reproducibility
    matrix = np.random.randint(1, 100, size=(num_nodes, num_nodes))
    np.fill_diagonal(matrix, 0)
    return (matrix + matrix.T) / 2  # Symmetric matrix

distance_matrix = create_distance_matrix(5)
colony = AntColonySystem(
    num_ants=10,
    num_nodes=5,
    alpha=1.0,
    beta=2.0,
    evaporation_rate=0.5,
    pheromone_deposit=10.0,
    iterations=50,
    distance_matrix=distance_matrix
)

best_path, best_length = colony.optimize()
print(f"Best Path: {best_path}, Best Length: {best_length}")