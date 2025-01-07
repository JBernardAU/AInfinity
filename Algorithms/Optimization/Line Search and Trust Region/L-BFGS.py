import numpy as np


class LBFGSSolver:
    """
    L-BFGS solver to minimize a function f(x), storing only the last m updates.

    Parameters
    ----------
    f : callable
        The objective function f(x) -> float.
    grad : callable
        The gradient of f, grad(x) -> np.ndarray of shape (n,).
    x0 : np.ndarray
        Initial guess for the solution, shape (n,).
    m : int
        The number of previous (s, y) pairs to store.
    tol : float, optional
        Tolerance for stopping (based on gradient norm).
    max_iter : int, optional
        Maximum number of iterations.
    alpha : float, optional
        Step size (learning rate). In practice, one would do a line search.
    """

    def __init__(self, f, grad, x0, m=5, tol=1e-6, max_iter=1000, alpha=1e-2):
        self.f = f
        self.grad = grad
        self.x = x0.astype(float)  # Ensure float
        self.m = m
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha

        # To store the (s_k, y_k) pairs
        self.s_history = []
        self.y_history = []
        self.rho_history = []

    def two_loop_recursion(self, q):
        """
        Compute B_k^(-1) q via the two-loop recursion, using the stored s_i, y_i.
        """
        alpha_vals = []
        # First loop (going backward)
        for s, y, rho in reversed(list(zip(self.s_history, self.y_history, self.rho_history))):
            alpha_i = rho * (s @ q)
            q -= alpha_i * y
            alpha_vals.append(alpha_i)

        # Typically, we scale by gamma_k = (y_{k-1}^T s_{k-1}) / (y_{k-1}^T y_{k-1})
        # if we want the best scaling of the identity. Let's do a simple scaling:
        if len(self.s_history) > 0:
            y_last = self.y_history[-1]
            s_last = self.s_history[-1]
            gamma_k = (s_last @ y_last) / (y_last @ y_last)
        else:
            gamma_k = 1.0

        r = gamma_k * q

        # Second loop (going forward)
        for (s, y, rho), alpha_i in zip(
                zip(self.s_history, self.y_history, self.rho_history),
                reversed(alpha_vals)
        ):
            beta = rho * (y @ r)
            r += (alpha_i - beta) * s

        return r

    def solve(self):
        """
        Perform L-BFGS iterations to find a local minimum of f(x).

        Returns
        -------
        x : np.ndarray
            The approximate minimizer.
        history : list of float
            The function values at each iteration (optional).
        """
        history = []

        g = self.grad(self.x)
        for i in range(self.max_iter):
            history.append(self.f(self.x))

            if np.linalg.norm(g) < self.tol:
                print(f"L-BFGS converged in {i} iterations.")
                break

            # Obtain the search direction p = -B_k^{-1} g using two-loop recursion
            p = -self.two_loop_recursion(g)

            # A simple step: x_new = x + alpha * p
            x_new = self.x + self.alpha * p

            # Compute s and y
            s = x_new - self.x
            g_new = self.grad(x_new)
            y = g_new - g

            # Update x and g
            self.x = x_new
            g = g_new

            # Check if y^T s > 0 (to avoid division by zero or negative)
            ys = y @ s
            if ys > 1e-14:
                rho = 1.0 / ys

                # If we exceed the history size, pop the oldest
                if len(self.s_history) == self.m:
                    self.s_history.pop(0)
                    self.y_history.pop(0)
                    self.rho_history.pop(0)

                # Store the new (s, y) pair
                self.s_history.append(s)
                self.y_history.append(y)
                self.rho_history.append(rho)

        return self.x, history


# Example usage
if __name__ == "__main__":
    # Again, let's use the 2D Rosenbrock function
    def f_rosenbrock(x):
        return (1 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


    def grad_rosenbrock(x):
        dfdx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2)
        dfdy = 200.0 * (x[1] - x[0] ** 2)
        return np.array([dfdx, dfdy], dtype=float)


    x0 = np.array([-1.2, 1.0])
    solver = LBFGSSolver(f_rosenbrock, grad_rosenbrock, x0, m=5, alpha=1e-3, max_iter=10000)
    x_opt, history = solver.solve()
    print("L-BFGS solution:", x_opt)
    print("f(x_opt) =", f_rosenbrock(x_opt))
