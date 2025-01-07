import numpy as np


class NewtonMethodSolver:
    """
    A simple Newton's Method solver for a single-variable equation f(x) = 0.

    Parameters
    ----------
    f : callable
        Function for which we want to find a root. f(x) should return a float.
    fprime : callable
        Derivative of f. fprime(x) should return a float.
    x0 : float, optional
        Initial guess for the root. Default is 0.0.
    tol : float, optional
        Tolerance for convergence. Iteration stops when the change in x
        is below this threshold, or when |f(x)| is below this threshold.
    max_iter : int, optional
        Maximum number of iterations to perform.
    """

    def __init__(self, f, fprime, x0=0.0, tol=1e-8, max_iter=100):
        self.f = f
        self.fprime = fprime
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter

    def solve(self):
        """
        Solve the equation f(x) = 0 using Newton's Method.

        Returns
        -------
        x : float
            The approximate root of the equation.
        """
        x = self.x0

        for i in range(self.max_iter):
            fx = self.f(x)
            dfx = self.fprime(x)

            # If the derivative is very small, we risk division by zero
            if abs(dfx) < 1e-14:
                print("Warning: Derivative too close to zero. Stopping iteration.")
                return x

            # Update step
            x_new = x - fx / dfx

            # Check convergence by function value or change in x
            if abs(self.f(x_new)) < self.tol or abs(x_new - x) < self.tol:
                x = x_new
                break

            x = x_new

        return x


if __name__ == "__main__":
    # Example usage: Solve for the root of f(x) = e^x - 2 = 0
    # The solution is x = ln(2) ~ 0.693147...

    def f_example(x):
        return np.exp(x) - 2


    def fprime_example(x):
        return np.exp(x)


    solver = NewtonMethodSolver(f=f_example, fprime=fprime_example, x0=1.0, tol=1e-12)
    root_approx = solver.solve()

    print("Approximate root using Newton's Method:", root_approx)
    print("Actual root (ln(2)):", np.log(2))
    print("Difference:", abs(root_approx - np.log(2)))
