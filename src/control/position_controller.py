from dataclasses import dataclass, field, fields
from typing import Tuple

import jax
import numpy as np
from mujoco_playground._src import mjx_env

from src.control.algorithms.base import ControllerParams
from src.control.algorithms.seq_pos_control import SequentialControllerParams
from src.control.controller_factory import ConfigFactory, ControllerFactory
from src.control.state import Go1State
from src.utils import load_dataclass_from_dict


@dataclass(kw_only=True)
class PositionControllerParams(ControllerParams):
    command_dim: int = field(default=3)
    primary_controller: ControllerParams = field(
        default_factory=lambda: SequentialControllerParams()  # or PolarCoordinateControllerParams()
    )
    fallback_controller: ControllerParams = field(default_factory=lambda: SequentialControllerParams())
    arrival_threshold: float = field(default=0.1)
    max_linear_velocity: float = field(default=1.5)
    max_angular_velocity: float = field(default=np.pi / 2)
    algorithm_type: str = field(default="position_controller")

    @classmethod
    def from_dict(cls, data: dict, convert_list_to_array=False):
        import copy

        config_factory = ConfigFactory()
        new_data = copy.deepcopy(data)  # Prevent modifying the original data
        new_data["primary_controller"] = config_factory.build(new_data["primary_controller"])
        new_data["fallback_controller"] = config_factory.build(new_data["fallback_controller"])
        return load_dataclass_from_dict(cls, new_data, convert_list_to_array=convert_list_to_array)


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

    def _calculate_reference_state(self, state: Go1State, target_pos: np.ndarray) -> Tuple[Go1State, Go1State]:
        target_yaw = self._compute_target_yaw(state, target_pos)
        target_yaw, current_yaw = np.unwrap([target_yaw, state.yaw])  # Unwrap angles to handle discontinuity
        state.yaw = current_yaw
        return state, Go1State(position=np.array([target_pos[0], target_pos[1], state.position[2]]), yaw=target_yaw)

    def _post_process_command(self, command: np.ndarray) -> jax.numpy.array:
        return jax.numpy.array(
            [
                np.clip(command[0], -self.max_linear_velocity, self.max_linear_velocity),
                0,
                np.clip(command[2], -self.max_angular_velocity, self.max_angular_velocity),
            ]
        )


def create_position_controller(
    controller_factory: ControllerFactory, config: PositionControllerParams
) -> PositionController:
    """
    Helper function to create PositionController

    Usage:
        config = PositionControllerParams(
            primary_controller=PolarCoordinateControllerParams(),
            fallback_controller=SequentialControllerParams(),
        )
        command_generator = create_position_controller(controller_factory=ControllerFactory(), config=config)
    """

    return PositionController(factory=controller_factory, config=config)
