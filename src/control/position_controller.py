from dataclasses import dataclass, field, fields
from typing import Tuple

import jax
import numpy as np
from mujoco_playground._src import mjx_env

from src.control.algorithms.base import Controller, ControllerParams
from src.control.controller_factory import ControllerFactory
from src.control.state import Go1State


@dataclass(kw_only=True)
class SequentialControllerParams(ControllerParams):
    yaw_control_threshold: float = field(default=np.pi / 18)
    yaw_control_gain: float = field(default=7.0)
    linear_control_gain: float = field(default=2.0)
    algorithm_type: str = field(default="seq_controller")


@dataclass(kw_only=True)
class PolarCoordinateControllerParams(ControllerParams):
    # make sure yaw_control_gain1 - linear_control_gain > yaw_control_gain2 for local stability
    linear_control_gain: float = field(default=2.0)
    yaw_control_gain1: float = field(default=7.0)
    yaw_control_gain2: float = field(default=1.0)
    rho_threshold: float = field(default=0.1)
    algorithm_type: str = field(default="polar_coord_controller")


@dataclass(kw_only=True)
class PositionControllerParams(ControllerParams):
    command_dim: int = field(default=3)
    primary_controller: ControllerParams = field(
        default_factory=lambda: SequentialControllerParams()  # or PolarCoordinateControllerParams()
    )
    fallback_controller: ControllerParams = field(
        default_factory=lambda: SequentialControllerParams()
    )
    arrival_threshold: float = field(default=0.1)
    max_linear_velocity: float = field(default=1.5)
    max_angular_velocity: float = field(default=np.pi / 2)
    algorithm_type: str = field(default="position_controller")


@dataclass(kw_only=True, frozen=True)
class PositionCommandInfo:
    pos: np.ndarray
    target_pos: np.ndarray
    is_arrived: bool


@dataclass(kw_only=True, frozen=True)
class PositionCommand:
    command: jax.Array
    info: PositionCommandInfo


class PositionController:
    """
    Position controller for the Go1 robot that generate velocity command to go to target XY position
    """

    def __init__(self, factory: ControllerFactory, config: PositionControllerParams | None = None):
        self.factory = factory
        self._controllers = {}

        config = config if config else PositionControllerParams()
        self.build_controller(config)

        self.prev_command = None

    def build_controller(self, config: PositionControllerParams) -> None:
        """
        Build controllers from configuration.
        If the item belongs to ControllerParams, build the controller using the factory and config.
        else, set the class attribute with the value.
        """
        for f in fields(config):
            key, value = f.name, getattr(config, f.name)
            if issubclass(type(value), ControllerParams):
                self._controllers[key] = self.factory.build(value)
            else:
                # Simple but unsafe dynamic attributiuon. No type safety and validation
                setattr(self, key, value)

    def compute_command(self, state: mjx_env.State, target_position: np.ndarray) -> PositionCommand:
        """
        Compute the velocity command [v_x, v_y, v_yaw] to reach the target position

        Args:
            state: Current state (from mjx env) of the robot
            target_position: Target position [x, y]

        Returns:
            command: velocity command to downsteam controller [linear_velocity, lateral_velocity, yaw_velocity]
            info: a dataclass with additional information
        """
        if self.prev_command is None:
            self.prev_command = jax.numpy.zeros(3)

        state = Go1State.from_mjx_state(state)
        state, ref_state = self._calculate_reference_state(state, target_position)
        dist = self._compute_distance(state, ref_state)

        if self._check_arrival(dist):
            command = jax.numpy.zeros(self.command_dim)
            info = self._create_return_info(state, ref_state, is_arrived=True)
            return PositionCommand(command=command, info=info)

        try:
            command = self._primary_control(state, ref_state)
        except Exception:
            command = self._fallback_control(state, ref_state)

        command = self._post_process_command(command)
        info = self._create_return_info(state, ref_state, is_arrived=False)
        return PositionCommand(command=command, info=info)

    def _primary_control(self, state: Go1State, ref_state: Go1State) -> np.ndarray:
        return self._controllers["primary_controller"].control(state, ref_state)

    def _fallback_control(self, state: Go1State, ref_state: Go1State) -> np.ndarray:
        return self._controllers["fallback_controller"].control(state, ref_state)

    def _compute_target_yaw(self, state: Go1State, target_pos: np.ndarray) -> float:
        """Compute desired yaw angle to face target."""
        return np.arctan2(target_pos[1] - state.position[1], target_pos[0] - state.position[0])

    def _compute_distance(self, state: Go1State, ref_state: Go1State) -> float:
        """Compute distance to target position."""
        return np.linalg.norm(state.position[:2] - ref_state.position[:2])

    def _check_arrival(self, distance: float) -> bool:
        """Check if robot has arrived at target position."""
        return distance <= self.arrival_threshold

    def _create_return_info(
        self,
        state: Go1State,
        ref_state: Go1State,
        is_arrived: bool,
    ) -> PositionCommandInfo:
        """Get additional information for the return value."""
        return PositionCommandInfo(
            pos=state.position,
            target_pos=ref_state.position,
            is_arrived=is_arrived,
        )

    def _calculate_reference_state(
        self, state: Go1State, target_pos: np.ndarray
    ) -> Tuple[Go1State, Go1State]:
        target_yaw = self._compute_target_yaw(state, target_pos)
        target_yaw, current_yaw = np.unwrap(
            [target_yaw, state.yaw]
        )  # Unwrap angles to handle discontinuity
        state.yaw = current_yaw
        return state, Go1State(
            position=np.array([target_pos[0], target_pos[1], state.position[2]]), yaw=target_yaw
        )

    def _post_process_command(self, command: np.ndarray) -> jax.numpy.array:
        return jax.numpy.array(
            [
                np.clip(command[0], -self.max_linear_velocity, self.max_linear_velocity),
                0,
                np.clip(command[2], -self.max_angular_velocity, self.max_angular_velocity),
            ]
        )


class SequentialController(Controller):
    """
    Simple rule-based proportional control
    """

    def __init__(self, config: SequentialControllerParams):
        self.params = config

    def control(self, state: Go1State, ref_state: Go1State, **kwargs) -> np.ndarray:
        x_error, y_error, yaw_error = (ref_state - state).to_array()
        dist = np.linalg.norm([x_error, y_error])

        if yaw_error > self.params.yaw_control_threshold:
            return np.array([0.0, 0.0, self.params.yaw_control_gain * yaw_error])
        else:
            return np.array([self.params.linear_control_gain * dist, 0.0, 0.0])


class PolarCoordinateController(Controller):
    """
    http://www.cs.cmu.edu/~rasc/Download/AMRobots3.pdf
    https://doi.org/10.1177/1729881418806435
    """

    def __init__(self, config: PolarCoordinateControllerParams):
        self.params = config

    def control(self, state: Go1State, ref_state: Go1State, **kwargs) -> np.ndarray:
        # Error calculation
        tar_yaw, yaw = np.unwrap([ref_state.yaw, state.yaw])
        error_x = ref_state.position[0] - state.position[0]
        error_y = ref_state.position[1] - state.position[1]
        error_yaw = tar_yaw - yaw

        # Polar coordinate transformation
        rho = np.sqrt(error_x**2 + error_y**2)
        alpha = np.arctan2(error_y, error_x) - yaw
        beta = alpha + yaw

        # Control inputs
        k1 = self.params.linear_control_gain
        k2 = self.params.yaw_control_gain1
        k3 = self.params.yaw_control_gain2
        v = k1 * rho * np.cos(alpha)
        if rho >= self.params.rho_threshold:  # prevent sigularity
            omega = k2 * alpha + k1 * (np.sin(alpha) * np.cos(alpha) / alpha) * (alpha + k3 * beta)
        else:
            omega = k2 * error_yaw
        return np.array([v, 0.0, omega])


def create_position_controller(
    controller_factory: ControllerFactory, config: PositionControllerParams
) -> PositionController:
    """
    Helper function to create PositionController
    """
    controller_factory.register_controller(SequentialControllerParams, SequentialController)
    controller_factory.register_controller(
        PolarCoordinateControllerParams, PolarCoordinateController
    )

    return PositionController(factory=controller_factory, config=config)
