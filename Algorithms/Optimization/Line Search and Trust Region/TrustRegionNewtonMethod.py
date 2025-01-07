import numpy as np


class TrustRegionNewtonSolver:
    """
    A Trust-Region Newton Method solver for unconstrained optimization.
    Uses the dogleg strategy to solve each trust-region subproblem.

    Parameters
    ----------
    f : callable
        The objective function f(x) -> float.
    grad : callable
        The gradient of f, grad(x) -> np.ndarray of shape (n,).
    hess : callable
        The Hessian of f, hess(x) -> np.ndarray of shape (n, n).
    x0 : np.ndarray
        Initial guess for the solution, shape (n,).
    Delta0 : float, optional
        Initial trust-region radius.
    eta1 : float, optional
        Lower threshold for the reduction ratio (e.g., 0.25).
    eta2 : float, optional
        Upper threshold for the reduction ratio (e.g., 0.75).
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for stopping (based on gradient norm).
    """

    def __init__(
            self,
            f,
            grad,
            hess,
            x0,
            Delta0=1.0,
            eta1=0.25,
            eta2=0.75,
            max_iter=200,
            tol=1e-6
    ):
        self.f = f
        self.grad = grad
        self.hess = hess
        self.x = x0.astype(float)  # ensure float
        self.Delta = Delta0
        self.eta1 = eta1
        self.eta2 = eta2
        self.max_iter = max_iter
        self.tol = tol

    def solve(self):
        """
        Perform the Trust-Region Newton steps to find a local minimum of f(x).

        Returns
        -------
        x : np.ndarray
            The approximate minimizer.
        history : list of float
            The function values at each iteration (optional).
        """
        history = []
        for i in range(self.max_iter):
            f_val = self.f(self.x)
            g = self.grad(self.x)
            H = self.hess(self.x)

            history.append(f_val)

            # Check convergence
            g_norm = np.linalg.norm(g)
            if g_norm < self.tol:
                print(f"Converged in {i} iterations, grad norm = {g_norm:.2e}.")
                break

            # Solve the trust-region subproblem using dogleg
            p = self._dogleg(g, H, self.Delta)

            # Compute actual reduction
            new_f_val = self.f(self.x + p)
            actual_red = f_val - new_f_val

            # Compute predicted reduction
            predicted_red = - (g @ p + 0.5 * p @ (H @ p))  # m_k(0) - m_k(p)

            # Compute reduction ratio r_k
            if predicted_red > 0:
                r = actual_red / predicted_red
            else:
                # This means our model predicted no improvement or negative improvement
                # Just set ratio to something small so we reduce Delta
                r = -1.0

            # Update trust region radius
            if r < self.eta1:
                # Shrink
                self.Delta *= 0.5
            else:
                # Accept the step
                self.x = self.x + p
                if r > self.eta2 and abs(np.linalg.norm(p) - self.Delta) < 1e-12:
                    # Step is on boundary and ratio is good, expand
                    self.Delta *= 2.0

        return self.x, history

    def _dogleg(self, g, H, Delta):
        """
        Dogleg method to find p within the trust region that approximately
        minimizes the quadratic model:
            m_k(p) = f(x_k) + g^T p + 0.5 p^T H p
        subject to ||p|| <= Delta.

        Returns the step p (np.ndarray).
        """
        # Compute the Newton step pN = - H^{-1} g (if H is invertible)
        # We use a robust solve here:
        try:
            pN = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # If Hessian is singular or near-singular, fallback to negative gradient
            # scaled to trust region boundary
            return - (Delta / np.linalg.norm(g)) * g

        # Compute the Cauchy (steepest descent) step pU
        # pU = -(g^T g)/(g^T H g) * g
        gHg = g @ (H @ g)
        if abs(gHg) < 1e-14:
            # If Hessian is near-singular in the direction of g,
            # fallback to scaled negative gradient
            pU = - (Delta / np.linalg.norm(g)) * g
        else:
            alpha = (g @ g) / gHg
            pU = -alpha * g

        norm_pN = np.linalg.norm(pN)
        norm_pU = np.linalg.norm(pU)

        if norm_pN <= Delta:
            # Full Newton step is inside the trust region
            return pN
        elif norm_pU >= Delta:
            # Cauchy point is outside the trust region, take scaled gradient
            return (Delta / norm_pU) * pU
        else:
            # We are in between, find alpha along the line from pU to pN
            pN_minus_pU = pN - pU
            # Solve || pU + alpha (pN - pU) || = Delta for alpha in [0, 1]
            # => ||pU||^2 + 2 alpha pU^T (pN - pU) + alpha^2 ||pN - pU||^2 = Delta^2
            # Solve quadratic for alpha
            a = pN_minus_pU @ pN_minus_pU
            b = 2.0 * pU @ pN_minus_pU
            c = pU @ pU - Delta ** 2

            # alpha >= 0
            alpha_candidates = []
            # Quadratic formula: alpha = (-b +/- sqrt(b^2 - 4ac)) / (2a)
            disc = b ** 2 - 4 * a * c
            if disc < 0:
                # Numerics gone weird or no real solution => fallback
                alpha = 1.0  # fallback
            else:
                sqrt_disc = np.sqrt(disc)
                alpha1 = (-b + sqrt_disc) / (2 * a)
                alpha2 = (-b - sqrt_disc) / (2 * a)
                # We want 0 <= alpha <= 1
                for cand in [alpha1, alpha2]:
                    if 0.0 <= cand <= 1.0:
                        alpha_candidates.append(cand)
                if len(alpha_candidates) == 0:
                    # fallback: pick alpha in [0,1] (closest)
                    alpha_candidates.append(min(max(alpha1, 0.0), 1.0))
                    alpha_candidates.append(min(max(alpha2, 0.0), 1.0))
                alpha = max(alpha_candidates)  # typically the positive root

            return pU + alpha * pN_minus_pU


if __name__ == "__main__":
    # Example: 2D Rosenbrock function
    def f_rosenbrock(x):
        return (1 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


    def grad_rosenbrock(x):
        dfdx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2)
        dfdy = 200.0 * (x[1] - x[0] ** 2)
        return np.array([dfdx, dfdy], dtype=float)


    def hess_rosenbrock(x):
        # Hessian of the Rosenbrock function
        d2fdx2 = 2.0 + 400.0 * (3.0 * x[0] ** 2 - x[1])
        d2fdy2 = 200.0
        d2fdxdy = -400.0 * x[0]
        return np.array([
            [d2fdx2, d2fdxdy],
            [d2fdxdy, d2fdy2]
        ], dtype=float)


    # Initial guess
    x0 = np.array([-1.2, 1.0])

    solver = TrustRegionNewtonSolver(
        f=f_rosenbrock,
        grad=grad_rosenbrock,
        hess=hess_rosenbrock,
        x0=x0,
        Delta0=1.0,
        max_iter=2000,
        tol=1e-8
    )

    x_opt, history = solver.solve()
    print("Trust-Region Newton solution:", x_opt)
    print("Final function value:", f_rosenbrock(x_opt))
    print("Number of iterations:", len(history))
