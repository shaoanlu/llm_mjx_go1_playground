from dataclasses import dataclass, field

import numpy as np
import scipy.linalg

from src.control.algorithms.base import Controller, ControllerParams


@dataclass(kw_only=True)  # Make all following fields keyword-only
class LQRParams(ControllerParams):
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray  # Diagonal elements of the quadratic cost matrix for the state variables
    R: np.ndarray  # Diagonal elements of the quadratic cost matrix for the control variables
    dt: float  # Time step
    algorithm_type: str = field(default="lqr")


class LQR(Controller):
    r"""
    A Linear Quadratic Regulator (LQR) controller that computes the optimal control input
    for a linear system with quadratic cost function.

    Model:
        \dot{x} = A * x + B * u
    Cost function:
        J = \int_0^{\infty} (x^T Q x + u^T R u) dt

    The optimal control input is given by:
        u = -K * x

    where K is the optimal feedback gain matrix obtained by solving the
    Riccati equation using scipy.linalg.solve_continuous_are.
    """

    def __init__(self, params: LQRParams):
        self.params = params
        self.A = params.A
        self.B = params.B
        self.Q = params.Q
        self.R = params.R

    def control(self, state: np.ndarray, ref_state: np.ndarray, **kwargs) -> np.ndarray:
        # Solve the continuous-time Algebraic Riccati Equation (ARE)
        P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)

        # Compute the optimal gain matrix K
        K = np.linalg.inv(self.R) @ self.B.T @ P
        state_error = state - ref_state
        command = -K @ state_error

        return command

    def update_model(
        self,
        A: np.ndarray | None = None,
        B: np.ndarray | None = None,
    ):
        self.A = A if A else self.A
        self.B = B if B else self.B

    def update_cost(
        self,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
    ):
        self.Q = Q if Q else self.Q
        self.R = R if R else self.R
