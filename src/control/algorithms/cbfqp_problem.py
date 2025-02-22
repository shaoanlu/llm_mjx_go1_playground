from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import proxsuite
from scipy import sparse


@dataclass(kw_only=True)
class QPProblemData:
    """Data structure for Quadratic Programming problem matrices and vectors."""

    P: sparse.csc_matrix  # Quadratic cost matrix
    q: np.ndarray  # Linear cost vector
    A: sparse.csc_matrix  # Constraint matrix
    l: np.ndarray  # Lower bounds vector
    u: np.ndarray  # Upper bounds vector


@dataclass(kw_only=True)
class CBFQPSolution:
    """Data structure for the solution of a Quadratic Programming problem for Control Barrier Functions (CBF)."""

    u: np.ndarray | None  # Optimal control input
    slack: np.ndarray | None  # Optimal slack variable
    qproblem: QPProblemData  # Original QP problem data


class CBFQPProblem:
    """
    Handles the formulation of Quadratic Programming (QP) problems for Control Barrier Functions (CBF).

                minimize  || u - u_nom ||^2 + k * δ^2
                    u, δ
                s.t.
                        h'(x, u) ≥ -𝛼 * h(x) - δ - d
                        u_min ≤ u ≤ u_max
                            0 ≤ δ ≤ inf
    where
        u = [ux, uy] is the control input in x and y axis respectively.
        u_nom is the nominal control input
        x is the state vector
        δ is the slack variable
        h(x) is the control barrier function and h'(x, u) its derivative
        d is the estimated disturbance of h'
        𝛼, is the CBF parameter
        k is the penalty coefficient for the slack variable
    """

    def __init__(self, nx: int, nh: int):
        """
        Initialize the QP problem formulation.

        Args:
            nx (int): Number of state variables (control dimensions)
            nh (int): Number of control barrier functions
        """
        self.nx = nx  # Number of states (control dimensions)
        self.nh = nh  # Number of barrier functions

    def create_matrices(
        self,
        h: List[float],
        coeffs_dhdx: List[List[float]],
        nominal_control: np.ndarray,
        max_control: np.ndarray,
        min_control: np.ndarray,
        cbf_alpha: float,
        slack_penalty: float,
        disturbance_h_dot: Optional[List[float]] = None,
        debug_mode: bool = False,
    ) -> QPProblemData:
        """
        Create all matrices and vectors needed for the QP problem.
        P: shape (nx, nx)
        q: shape (nx,)
        A: shape (nx+nh, nx)
        l: shape (nh+nx,)
        u: shape (nh+nx,)
        nx: number of state (control dim)
        nh: number of control barrier functions (equals to number of slack variables)

        The QP problem is formulated as:
            minimize 1/2 * x^T * P * x + q^T * x
            s.t. l <= A * x <= u

        Args:
            h (List[float]): Barrier function values
            coeffs_dhdx (List[List[float]]): Coefficients of barrier function derivatives
            nominal_control (np.ndarray): Nominal control inputs
            max_control (np.ndarray): Maximum allowed control values
            min_control (np.ndarray): Minimum allowed control values
            cbf_alpha (float): CBF constraint parameter
            slack_penalty (float): Penalty coefficient for slack variables
            disturbance_h_dot (Optional[List[float]]): Disturbance terms for barrier functions

        Returns:
            QPProblemData: Container with all matrices and vectors for the QP problem
        """
        if debug_mode:
            self._validate_input(h, coeffs_dhdx, nominal_control, max_control, min_control, cbf_alpha, slack_penalty)

        if disturbance_h_dot is None:
            disturbance_h_dot = [0.0] * self.nh

        P = self._create_cost_matrix(slack_penalty)
        q = self._create_cost_vector(nominal_control)
        A = self._create_constraint_matrix(coeffs_dhdx)
        l, u = self._create_bound_vectors(h, disturbance_h_dot, cbf_alpha, max_control, min_control)

        return QPProblemData(P=P, q=q, A=A, l=l, u=u)

    def solve(self, data: QPProblemData) -> CBFQPSolution:
        # ProxQP uses different notations for the QP problemu
        # As solving QP problem is intuitive with ProxQP, we can directly use it w/o extra abstraction
        res = proxsuite.proxqp.dense.solve(H=data.P, g=data.q, C=data.A, l=data.l, u=data.u)
        if res.x is None:
            CBFQPSolution(u=None, slack=None, qproblem=data)
        opt_u = res.x[: self.nx]
        opt_slack = res.x[self.nx :]
        return CBFQPSolution(u=opt_u, slack=opt_slack, qproblem=data)

    def _create_cost_matrix(self, slack_penalty: float) -> sparse.csc_matrix:
        """
        Create the quadratic cost matrix P.

        Args:
            slack_penalty (float): Penalty coefficient for slack variables

        Returns:
            sparse.csc_matrix: Sparse matrix P of shape (nx + nh, nx + nh)
        """
        P = np.eye(self.nx + self.nh)
        P[self.nx :, self.nx :] = P[self.nx :, self.nx :] * slack_penalty  # Penalize slack variables
        return P

    def _create_cost_vector(self, nominal_control: np.ndarray) -> np.ndarray:
        """
        Create the linear cost vector q.

        Args:
            nominal_control (np.ndarray): Nominal control inputs

        Returns:
            np.ndarray: Vector q of shape (nx + nh,)
        """
        return np.hstack([-nominal_control, np.zeros(self.nh)])  # Minimize deviation from nominal control

    def _create_constraint_matrix(self, coeffs_dhdx: List[List[float]]) -> sparse.csc_matrix:
        """
        Create the constraint matrix A.

        Args:
            coeffs_dhdx (List[List[float]]): Coefficients of barrier function derivatives ∇h(x)

        Returns:
            sparse.csc_matrix: Matrix A of shape (nh + nx + nh, nh + nx)
        """
        cbf_constraint_matrix = np.array(coeffs_dhdx)
        control_slack_constraints_matrix = np.eye(self.nx + self.nh)
        A = np.vstack([cbf_constraint_matrix, control_slack_constraints_matrix])
        return A

    def _create_bound_vectors(
        self,
        h: List[float],
        disturbance_h_dot: List[float],
        cbf_alpha: float,
        max_control: np.ndarray,
        min_control: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create the lower and upper bound vectors l and u that abide by the cbf constraints:

            h_dot + disturbance_h_dot >= -alpha * h
            u_min <= u <= u_max
            -inf <= δ <= inf

        Args:
            h (List[float]): Barrier function values
            disturbance_h_dot (List[float]): Disturbance terms for barrier functions
            cbf_alpha (float): CBF constraint parameter
            max_control (np.ndarray): Maximum allowed control values
            min_control (np.ndarray): Minimum allowed control values

        Returns:
            Tuple[np.ndarray, np.ndarray]: Lower and upper bound vectors
        """
        # Lower bounds: CBF constraints and control/slack bounds
        lb_cbf_constr = np.array([-cbf_alpha * h_ - d_ for h_, d_ in zip(h, disturbance_h_dot)])
        lb_control_constr = min_control
        lb_slack_constr = np.full(self.nh, -np.inf)
        lb = np.concatenate([lb_cbf_constr, lb_control_constr, lb_slack_constr])

        # Upper bounds: Infinity for CBF constraints, control bounds, and slack bounds
        ub_cbf_constr = np.full(self.nh, np.inf)
        ub_control_constr = max_control
        ub_slack_constr = np.full(self.nh, np.inf)
        ub = np.concatenate([ub_cbf_constr, ub_control_constr, ub_slack_constr])

        return lb, ub

    def _validate_input(
        self,
        h: List[float],
        coeffs_dhdx: List[List[float]],
        nominal_control: np.ndarray,
        max_control: np.ndarray,
        min_control: np.ndarray,
        cbf_alpha: float,
        slack_penalty: float,
    ):
        assert len(h) == self.nh, (
            f"Barrier function values must match the number of barrier functions, {h=}, {self.nh=}"
        )
        assert len(coeffs_dhdx) == self.nh, (
            f"Barrier function derivative coefficients must match the number of barrier functions, "
            f"{coeffs_dhdx=}, {self.nh=}"
        )
        for i in range(self.nh):
            assert len(coeffs_dhdx[i]) == self.nx + self.nh, (
                f"Barrier function derivative coefficients must match the number of states and slack variables, "
                f"{coeffs_dhdx=}, {self.nx=}"
            )
        assert nominal_control.shape == (self.nx,), (
            f"Nominal control input must have shape (nx,), {nominal_control.shape=}, {self.nx=}"
        )
        assert max_control.shape == (self.nx,), (
            f"Maximum control input must have shape (nx,), {max_control.shape=}, {self.nx=}"
        )
        assert min_control.shape == (self.nx,), (
            f"Minimum control input must have shape (nx,), {min_control.shape=}, {self.nx=}"
        )
        assert cbf_alpha > 0, f"CBF constraint parameter must be positive, {cbf_alpha=}"
        assert slack_penalty >= 0, f"Slack penalty must be non-negative, {slack_penalty=}"
