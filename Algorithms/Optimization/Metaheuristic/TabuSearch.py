import random
import math


class TSPTabuSearch:
    """
    A class implementing Tabu Search for the Traveling Salesman Problem (TSP).

    Attributes:
        distance_matrix (list[list[float]]): A 2D matrix of distances between cities.
        max_iterations (int): Maximum number of iterations to run the search.
        tabu_tenure (int): Number of iterations that a move remains tabu.
        seed (int, optional): Seed for random number generator (for reproducibility).

    Usage:
        ts = TSPTabuSearch(distance_matrix, max_iterations=100, tabu_tenure=5)
        best_route, best_cost = ts.run()
    """

    def __init__(self, distance_matrix, max_iterations=100, tabu_tenure=5, seed=None):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure

        # Optional: Set a random seed for reproducibility
        if seed is not None:
            random.seed(seed)

        # Will store moves (i, j) -> iteration when tabu expires
        self.tabu_list = {}

    def run(self):
        """
        Perform the Tabu Search and return the best route found and its cost.
        """
        # Generate an initial solution (random permutation of cities)
        current_solution = self._generate_initial_solution()
        current_cost = self._calculate_route_cost(current_solution)

        # Keep track of the best solution
        best_solution = current_solution
        best_cost = current_cost

        for iteration in range(1, self.max_iterations + 1):
            best_candidate = None
            best_candidate_cost = math.inf
            best_move = None

            # Generate neighbors by swapping each pair (i, j)
            for i in range(self.num_cities):
                for j in range(i + 1, self.num_cities):
                    # The 'move' is swapping the positions i and j in the route
                    move = (min(i, j), max(i, j))  # store consistently

                    candidate_solution = self._swap_cities(current_solution, i, j)
                    candidate_cost = self._calculate_route_cost(candidate_solution)

                    # Check if move is tabu
                    is_tabu = self._is_move_tabu(move, iteration)

                    # Aspiration criterion: override if it improves the global best
                    if not is_tabu or candidate_cost < best_cost:
                        if candidate_cost < best_candidate_cost:
                            best_candidate = candidate_solution
                            best_candidate_cost = candidate_cost
                            best_move = move

            # If we found any valid candidate, update our current solution
            if best_candidate is not None:
                current_solution = best_candidate
                current_cost = best_candidate_cost

                # Update global best if improved
                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost

                # Mark the chosen move as tabu with expiration
                self.tabu_list[best_move] = iteration + self.tabu_tenure

            # Remove any expired moves from the tabu list
            self._remove_expired_moves(iteration)

            # (Optional) Uncomment for debug / progress output:
            # print(f"Iteration {iteration}, Current Cost: {current_cost}, Best Cost: {best_cost}")

        return best_solution, best_cost

    # -----------------------
    # Internal Helper Methods
    # -----------------------

    def _generate_initial_solution(self):
        """
        Generate a random initial permutation of the cities.
        """
        route = list(range(self.num_cities))
        random.shuffle(route)
        return route

    def _calculate_route_cost(self, route):
        """
        Calculate the total travel cost of the given route,
        returning to the start city at the end.
        """
        total_cost = 0
        for i in range(len(route)):
            current_city = route[i]
            next_city = route[(i + 1) % len(route)]  # wrap around
            total_cost += self.distance_matrix[current_city][next_city]
        return total_cost

    def _swap_cities(self, route, i, j):
        """
        Return a new route with the cities at positions i and j swapped.
        """
        new_route = list(route)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route

    def _is_move_tabu(self, move, current_iteration):
        """
        Check if the move (i, j) is still in the tabu list
        for the current iteration.
        """
        # Move is tabu if it's in the list and expires after the current iteration.
        return move in self.tabu_list and self.tabu_list[move] > current_iteration

    def _remove_expired_moves(self, current_iteration):
        """
        Remove moves from the tabu list that have expired
        (i.e., their expiration iteration <= current_iteration).
        """
        expired_moves = [m for m, expiry in self.tabu_list.items() if expiry <= current_iteration]
        for move in expired_moves:
            del self.tabu_list[move]


# ----------------------------
# Example usage (main check)
# ----------------------------
if __name__ == "__main__":
    # Example distance matrix (4 cities)
    distance_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]

    # Create an instance of TSPTabuSearch
    tabu_search = TSPTabuSearch(distance_matrix, max_iterations=100, tabu_tenure=5, seed=42)

    # Run the search
    best_route, best_cost = tabu_search.run()

    print("Best route found:", best_route)
    print("Best route cost:", best_cost)
