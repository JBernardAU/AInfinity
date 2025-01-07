import numpy as np


class BFGSSolver:
    """
    BFGS solver to minimize a function f(x).

    Parameters
    ----------
    f : callable
        The objective function f(x) -> float.
    grad : callable
        The gradient of f, grad(x) -> np.ndarray of shape (n,).
    x0 : np.ndarray
        Initial guess for the solution, shape (n,).
    tol : float, optional
        Tolerance for stopping (based on gradient norm or step size).
    max_iter : int, optional
        Maximum number of iterations.
    alpha : float, optional
        Step size (learning rate). For robust convergence,
        a proper line search is typically used, but we keep it simple here.
    """

    def __init__(self, f, grad, x0, tol=1e-6, max_iter=1000, alpha=1e-2):
        self.f = f
        self.grad = grad
        self.x = x0.astype(float)  # Ensure float
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha

    def solve(self):
        """
        Perform BFGS iterations to find a local minimum of f(x).

        Returns
        -------
        x : np.ndarray
            The approximate minimizer.
        history : list of float
            The function values at each iteration (optional).
        """
        n = len(self.x)
        B = np.eye(n)  # Initial Hessian approximation
        history = []

        for i in range(self.max_iter):
            g = self.grad(self.x)
            history.append(self.f(self.x))

            # Check for convergence based on gradient norm
            if np.linalg.norm(g) < self.tol:
                print(f"Converged in {i} iterations.")
                break

            # Compute the direction p = -B^{-1} * g by solving B p = g
            # But we already have B, not B^{-1}. So let's solve for p:
            p = np.linalg.solve(B, -g)

            # A simple step: x_new = x + alpha * p
            # In practice, do a line search to find a good alpha
            x_new = self.x + self.alpha * p
            s = x_new - self.x  # s_k

            # Compute new gradient and y
            g_new = self.grad(x_new)
            y = g_new - g  # y_k

            # Update B if y^T s != 0 to avoid division by zero
            ys = y @ s
            if abs(ys) > 1e-14:
                Bs = B @ s
                B = B - np.outer(Bs, Bs) / (s @ Bs) + np.outer(y, y) / ys

            # Update x
            self.x = x_new

        return self.x, history


# Example usage
if __name__ == "__main__":
    # Let's minimize the 2D Rosenbrock function, which is a common test problem:
    # f(x, y) = (1 - x)^2 + 100 (y - x^2)^2
    # The global minimum is at x = (1, 1).

    def f_rosenbrock(x):
        return (1 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


    def grad_rosenbrock(x):
        dfdx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2)
        dfdy = 200.0 * (x[1] - x[0] ** 2)
        return np.array([dfdx, dfdy], dtype=float)


    x0 = np.array([-1.2, 1.0])
    solver = BFGSSolver(f_rosenbrock, grad_rosenbrock, x0, alpha=1e-3, max_iter=10000)
    x_opt, history = solver.solve()
    print("BFGS solution:", x_opt)
    print("f(x_opt) =", f_rosenbrock(x_opt))
