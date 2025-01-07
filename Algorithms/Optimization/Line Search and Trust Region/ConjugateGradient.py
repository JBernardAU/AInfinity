import numpy as np


class ConjugateGradientSolver:
    """
    A simple Conjugate Gradient (CG) solver for the system A x = b.

    Parameters
    ----------
    A : numpy.ndarray
        A symmetric positive-definite matrix of shape (n, n).
    b : numpy.ndarray
        Right-hand side vector of shape (n,).
    x0 : numpy.ndarray, optional
        Initial guess for the solution. If None, a zero vector is used.
    tol : float, optional
        Tolerance for convergence. The algorithm stops when the norm
        of the residual is below this threshold.
    max_iter : int, optional
        Maximum number of iterations. If None, default is the size of b.
    """

    def __init__(self, A, b, x0=None, tol=1e-8, max_iter=None):
        self.A = A
        self.b = b
        self.n = b.shape[0]

        if x0 is None:
            self.x0 = np.zeros_like(b)
        else:
            self.x0 = x0

        self.tol = tol
        self.max_iter = self.n if max_iter is None else max_iter

    def solve(self):
        """
        Solve the system A x = b using the Conjugate Gradient method.

        Returns
        -------
        x : numpy.ndarray
            The approximate solution vector.
        """
        x = self.x0.copy()
        r = self.b - self.A @ x  # Residual
        p = r.copy()  # Search direction
        rr_old = r @ r  # Dot product of r with itself

        # Main loop
        for i in range(self.max_iter):
            Ap = self.A @ p
            alpha = rr_old / (p @ Ap)
            x += alpha * p
            r -= alpha * Ap

            # Check convergence
            rr_new = r @ r
            if np.sqrt(rr_new) < self.tol:
                break

            beta = rr_new / rr_old
            p = r + beta * p
            rr_old = rr_new

        return x


if __name__ == "__main__":
    # Example usage:
    # Suppose we want to solve the system A x = b for a symmetric positive-definite A.
    n = 5
    np.random.seed(0)

    # Generate a random symmetric positive-definite matrix
    Q = np.random.randn(n, n)
    A = Q.T @ Q + np.eye(n)  # ensures positive-definite
    b = np.random.randn(n)

    # Create the solver and solve
    solver = ConjugateGradientSolver(A, b, tol=1e-8)
    x_cg = solver.solve()

    # Compare to NumPy's direct solve
    x_direct = np.linalg.solve(A, b)

    print("Conjugate Gradient solution:", x_cg)
    print("Direct solve solution:      ", x_direct)
    print("Difference norm:           ", np.linalg.norm(x_cg - x_direct))
