from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.special import logsumexp

from src.control.algorithms.base import ControllerParams, HighLevelController
from src.control.algorithms.cbfqp_problem import CBFQPProblem, CBFQPSolution, QPProblemData
from src.control.models import ControlAffineSystem, Simple2DRobot
from src.control.state import Go1State


@dataclass(kw_only=True)
class SafetyFilterParams(ControllerParams):
    model: ControlAffineSystem = field(default_factory=lambda: Simple2DRobot())
    max_output: Tuple = field(default=(1.2, 0.7))
    min_output: Tuple = field(default=(-1.2, -0.7))
    cbf_alpha: float = field(default=1.0)
    cbf_slack_penalty: float = field(default=10.0)
    cbf_kappa: float = field(default=0.5)
    algorithm_type: str = field(default="safety_filter")

    def __post_init__(self):
        # Validate the parameters
        assert self.cbf_alpha > 0, f"CBF alpha must be positive, {self.cbf_alpha=}"
        assert self.cbf_kappa > 0, f"CBF kappa must be positive, {self.cbf_kappa=}"
        assert self.cbf_slack_penalty > 0, f"CBF slack penalty must be positive, {self.cbf_slack_penalty=}"
        assert len(self.max_output) == len(self.min_output), (
            f"Max output and min output must have the same dimensions, {self.max_output=}, {self.min_output=}"
        )
        assert np.all(self.min_output <= self.max_output), (
            f"Min output must be less than max output, {self.min_output=}, {self.max_output=}"
        )


@dataclass(kw_only=True)
class SafeCommand:
    command: np.ndarray | None
    info: CBFQPSolution | None  # None if no need to compute safe command when there are no obstacles


class SafetyFilter(HighLevelController):
    """
    Safety filter controller based on Control Barrier Functions (CBF).
    The safety filter is a high-level controller that modifies the nominal velocity command for the Go1 robot
    to avoid collision with obstacles.
    """

    def __init__(self, config: SafetyFilterParams):
        self.build_controller(config)

    def build_controller(self, config: SafetyFilterParams):
        self.config = config
        self.model = self.config.model
        self.nx = self._get_input_dim()
        self.nh = self._get_barrier_dim()
        self.max_control, self.min_control = self._get_control_bounds()

    def _get_input_dim(self):
        """Return number of input dimensions."""
        return self.model.u_dim

    def _get_barrier_dim(self):
        """Return number of barrier function: Fixed as 1 as the safety filter uses a composite CBF"""
        return 1

    def _get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.config.max_output), np.array(self.config.min_output)

    def compute_command(
        self, state: Go1State, command: np.ndarray, obstacle_positions: List[np.ndarray], **kwargs
    ) -> SafeCommand:
        """
        Compute the safety filter command that does minimal modifications to the nominal command while ensuring safety.
        Args:
            state: The current state of the system.
            command: The nominal command to be modified.
            obstacle_positions: The positions of the obstacles in the environment.

        Returns:
            The safety filtered command.

        NOTE: approximate [v, w] to [v_linear, v_lateral] for Go1 robot
            v_linear = v
            v_lateral = w * self.model.a
        """
        if len(obstacle_positions) == 0:
            return SafeCommand(command=command, info=None)

        # Preprocess and validate the input
        obstacle_positions = np.array(obstacle_positions)
        state = self.model.preprocess_go1_state(state)
        self._validate_input(state=state, command=command, obstacle_positions=obstacle_positions)

        # Calculate the barrier function and its derivative coefficients
        h, dhdx = self._calculate_composite_cbf_coeffs(pos=state, obs_pos=obstacle_positions)

        # Solve the CBF-QP problem
        prob = CBFQPProblem(nx=self.nx, nh=self.nh)
        qp_data: QPProblemData = prob.create_matrices(
            h=h,
            coeffs_dhdx=dhdx,
            nominal_control=command,
            max_control=self.max_control,
            min_control=self.min_control,
            cbf_alpha=self.config.cbf_alpha,
            slack_penalty=self.config.cbf_slack_penalty,
            disturbance_h_dot=self._estimate_disturbance(),
        )
        sol: CBFQPSolution = prob.solve(qp_data)

        return SafeCommand(command=sol.u, info=sol)

    def _validate_input(self, state: np.ndarray, command: np.ndarray, obstacle_positions: np.ndarray):
        assert state.shape == (self.model.x_dim,), (
            f"current position pos shape must match the state dimension, {state.shape=}, {self.model.x_dim=}"
        )
        assert command.shape == (self.model.u_dim,), (
            f"command shape must match the control dimension, {command.shape=}, {self.model.u_dim=}"
        )
        assert state.shape == (obstacle_positions.shape[1],), (
            f"obs_pos shape must match the state dimension, {obstacle_positions.shape=}, {state.shape=}"
        )

    def _calculate_composite_cbf_coeffs(
        self, pos: np.ndarray, obs_pos: np.ndarray
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Calculate the barrier function and its derivative coefficients.
        main operation in this function is to calculate the composite barrier function that
        combines multiple barrier functions into a single barrier function.

        References:
            - Harms, Marvin, Martin Jacquet, and Kostas Alexis. "Safe Quadrotor Navigation using Composite Control Barrier Functions." arXiv preprint arXiv:2502.04101 (2025).
            - Molnar, Tamas G., and Aaron D. Ames. "Composing control barrier functions for complex safety specifications." IEEE Control Systems Letters (2023).

        Args:
            pos: The current position of the robot.
            obs_pos: The positions of the obstacles in the environment.

        Returns:
            The composite barrier value and its derivative coefficients.
        """

        # Calculate the barrier term
        hi_x = self.model.h(x=pos, obs_x=obs_pos)  # shape=(nh,)
        h_x = -1 / self.config.cbf_kappa * logsumexp(-self.config.cbf_kappa * hi_x)  # compisite hi(x) to shape=(1,)
        assert hi_x.shape == (len(obs_pos),), (
            f"hi_x shape must match the number of obstacles, {hi_x.shape=}, {obs_pos.shape=}"
        )

        # Calculate the derivative of the composite barrier term
        composite_weights = np.exp(-self.config.cbf_kappa * (hi_x - h_x)).reshape(-1, 1)  # shape=(nh, 1)
        h_dot = self.model.h_dot(x=pos, obs_x=obs_pos)  # shape=(nh, nx)
        dhdx = np.sum(composite_weights * h_dot, axis=0).reshape(-1)  # sum over obstacles, shape=(nx,)
        assert dhdx.shape == (self.nx,), f"dhdx shape must match the input dimension, {dhdx.shape=}, {self.nx=}"
        return [h_x], [list(dhdx) + [1.0] * self.nh]

    def _estimate_disturbance(self, **kwargs) -> List[float] | None:
        # TODO: implement disturbance observer
        return None
