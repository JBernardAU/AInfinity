import numpy as np

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = self.position.copy()
        self.best_value = float('inf')
        self.current_value = float('inf')

    def update_velocity(self, global_best_position, inertia, cognitive, social):
        r1 = np.random.uniform(0, 1, len(self.position))
        r2 = np.random.uniform(0, 1, len(self.position))

        cognitive_component = cognitive * r1 * (self.best_position - self.position)
        social_component = social * r2 * (global_best_position - self.position)
        self.velocity = inertia * self.velocity + cognitive_component + social_component

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])

class ParticleSwarmOptimizer:
    def __init__(self, objective_function, dim, bounds, num_particles=30, inertia=0.7, cognitive=1.5, social=1.5, max_iter=100):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.max_iter = max_iter

        self.particles = [Particle(dim, bounds) for _ in range(num_particles)]
        self.global_best_position = np.random.uniform(bounds[0], bounds[1], dim)
        self.global_best_value = float('inf')

    def optimize(self):
        for iteration in range(self.max_iter):
            for particle in self.particles:
                particle.current_value = self.objective_function(particle.position)

                if particle.current_value < particle.best_value:
                    particle.best_value = particle.current_value
                    particle.best_position = particle.position.copy()

                if particle.current_value < self.global_best_value:
                    self.global_best_value = particle.current_value
                    self.global_best_position = particle.position.copy()

            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.inertia, self.cognitive, self.social)
                particle.update_position(self.bounds)

            print(f"Iteration {iteration + 1}/{self.max_iter}, Best Value: {self.global_best_value}")

        return self.global_best_position, self.global_best_value

# Example usage
def objective_function(x):
    return np.sum(x**2)  # Example: minimize sum of squares

bounds = [-10, 10]  # Search space bounds
optimizer = ParticleSwarmOptimizer(objective_function, dim=2, bounds=bounds, max_iter=50)
best_position, best_value = optimizer.optimize()
print(f"Best Position: {best_position}, Best Value: {best_value}")