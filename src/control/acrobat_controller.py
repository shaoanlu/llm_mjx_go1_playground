from enum import Enum, auto
from typing import Any, Dict

import jax
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from src.control.algorithms.base import Controller
from src.control.algorithms.mlp import MLPPolicy, MLPPolicyParams
from src.control.controller_factory import ConfigFactory, ControllerFactory


JOYSTICK_ENV_ACTION_SCALE = 0.5
HANDSTAND_ENV_ACTION_SCALE = 0.3
GETUP_ENV_ACTION_SCALE = 0.5
JOYSTICK_ENV_DEFAULT_POSE = jax.numpy.array([0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8])


class Go1ControllerType(Enum):
    """Available controller types."""

    JOYSTICK = auto()
    HANDSTAND = auto()
    FOOTSTAND = auto()
    GETUP = auto()


class MLPPolicyJoystick2HandstandAdapter(Controller):
    """
    MLPPolicy controller trained in Joystick env with necessary state and action adaptations to Handstand env
    In Joystick env:
        # self._default_pose: Array([ 0.1,  0.9, -1.8, -0.1,  0.9, -1.8,  0.1,  0.9, -1.8, -0.1,  0.9, -1.8], dtype=float32)
        motor_targets = self._default_pose + action * 0.5
    In Handstand env:
        motor_targets = stata.data.ctrl + action * 0.3

    This controller performs the necessary adaptations in .control() to the state and action
    """

    def __init__(self, controller: Controller):
        self._controller = controller
        self._src_env_action_scale = JOYSTICK_ENV_ACTION_SCALE
        self._tar_env_action_scale = HANDSTAND_ENV_ACTION_SCALE
        self._src_default_pose = JOYSTICK_ENV_DEFAULT_POSE

    def control(self, state: mjx_env.State, command: np.ndarray, mjx_state_data: mjx.Data) -> np.ndarray:
        """Control with state and action space adaptation."""
        # Adapt state for joystick control
        state = jax.numpy.concat([state, command])

        # Get control action
        action: np.ndarray = self._controller.control(state)

        # Adapt action space
        action = (
            self._src_env_action_scale * action - mjx_state_data.ctrl + self._src_default_pose
        ) / self._tar_env_action_scale

        return action


class MLPPolicyGetup2HandstandAdapter(Controller):
    """
    MLPPolicy controller trained in Getup env with necessary state and action adaptations to Handstand env
    In Getup env:
        motor_targets = state.data.qpos[7:] + action * 0.5
    In Handstand env:
        motor_targets = stata.data.ctrl + action * 0.3

    This controller performs the necessary adaptations in .control() to the state and action
    """

    def __init__(self, controller: Controller):
        self._controller = controller
        self._src_env_action_scale = GETUP_ENV_ACTION_SCALE
        self._tar_env_action_scale = HANDSTAND_ENV_ACTION_SCALE

    def control(self, state: mjx_env.State, command: np.ndarray, mjx_state_data: mjx.Data) -> np.ndarray:
        """Control with state and action space adaptation."""
        # Adapt state for Getup control
        state = state[3:]  # remove first 3 linvel elements

        # Get control action
        action: np.ndarray = self._controller.control(state)

        # Adapt action space
        action = (
            self._src_env_action_scale * action - mjx_state_data.ctrl + mjx_state_data.qpos[7:]
        ) / self._tar_env_action_scale

        return action


class Go1ControllerManager:
    """Manages multiple controllers and handles transitions between them."""

    def __init__(self, controllers: Dict[Go1ControllerType, Controller]):
        self._controllers = controllers
        self._active_type = Go1ControllerType.FOOTSTAND  # default controller
        self._command = jax.numpy.zeros(3)  # joystick command (vel_x, vel_y, vel_yaw)

    def set_command(self, command: jax.Array | np.ndarray):
        """Set the current command for joystick controller."""
        if command.shape != np.empty((3,)).shape:
            raise ValueError(f"Invalid command shape {command.shape}. Expected (3,)")
        self._command = command

    def switch_controller(self, controller_type: Go1ControllerType):
        """Switch to a different controller type."""
        if controller_type not in self._controllers:
            raise ValueError(f"No controller registered for type {controller_type}")
        self._active_type = controller_type

    def control(self, state: mjx_env.State) -> np.ndarray:
        """Get control action from current active controller."""
        controller = self._controllers[self._active_type]
        return controller.control(state.obs["state"], command=self._command, mjx_state_data=state.data)


def create_acrobat_controller_manager(
    controller_factory: ControllerFactory,
    config_factory: ConfigFactory,
    controller_configs: Dict[Go1ControllerType, Dict[str, Any]],
) -> Go1ControllerManager:
    """
    Create a configured Go1ControllerManager.

    NOTE:
    given the use case for the acrobatic controller being quite flexible
    It feels more confortable to have the creation of controller manager
    defined in a funciton instead of from a config file
    """

    controllers = {}

    # Create each controller
    config_factory.register_config("mlp", MLPPolicyParams)
    controller_factory.register_controller(MLPPolicyParams, MLPPolicy)
    for controller_type, config in controller_configs.items():
        params = config_factory.build(config)
        base_controller = controller_factory.build(params=params)

        # Wrap joystick and getup controller with adapter and leave others as is
        # Regarding the adapter, refer to MLPPolicyEnvName2HandstandAdapter for more details
        if controller_type == Go1ControllerType.JOYSTICK:
            controllers[controller_type] = MLPPolicyJoystick2HandstandAdapter(controller=base_controller)
        elif controller_type == Go1ControllerType.GETUP:
            controllers[controller_type] = MLPPolicyGetup2HandstandAdapter(controller=base_controller)
        else:
            controllers[controller_type] = base_controller

    return Go1ControllerManager(controllers)
