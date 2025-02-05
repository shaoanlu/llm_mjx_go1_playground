from enum import Enum, auto
from typing import Any, Dict

import jax
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from examples.mujoco_Go1.env_wrapper import Go1Env
from examples.mujoco_Go1.ppo import PPO, PPOParams, PPOParamsBuilder
from src.control.algorithm.base import Controller
from src.control.controller_factory import ControllerFactory


class Go1ControllerType(Enum):
    """Available controller types."""

    JOYSTICK = auto()
    HANDSTAND = auto()
    FOOTSTAND = auto()
    GETUP = auto()


class PPOJoystick2HandstandAdapter(Controller):
    """
    PPO controller trained in Joystick env with necessary state and action adaptations to Handstand env
    In Joystick env:
        # self._default_pose: Array([ 0.1,  0.9, -1.8, -0.1,  0.9, -1.8,  0.1,  0.9, -1.8, -0.1,  0.9, -1.8], dtype=float32)
        motor_targets = self._default_pose + action * 0.5
    In Handstand env:
        motor_targets = stata.data.ctrl + action * 0.3

    This controller performs the necessary adaptations in .control() to the state and action
    """

    def __init__(self, controller: Controller, joystick_env: Any, handstand_env: Any):
        self._controller = controller
        self._src_env_action_scale = joystick_env.env_cfg.action_scale
        self._tar_env_action_scale = handstand_env.env_cfg.action_scale
        self._src_default_pose = joystick_env.env._default_pose

    def control(self, state: mjx_env.State, command: np.ndarray, data: mjx.Data) -> np.ndarray:
        """Control with state and action space adaptation."""
        # Adapt state for joystick control
        state = jax.numpy.concat([state, command])

        # Get control action
        action: np.ndarray = self._controller.control(state)

        # Adapt action space
        action = (
            self._src_env_action_scale * action - data.ctrl + self._src_default_pose
        ) / self._tar_env_action_scale

        return action


class PPOGetup2HandstandAdapter(Controller):
    """
    PPO controller trained in Getup env with necessary state and action adaptations to Handstand env
    In Getup env:
        motor_targets = state.data.qpos[7:] + action * 0.5
    In Handstand env:
        motor_targets = stata.data.ctrl + action * 0.3

    This controller performs the necessary adaptations in .control() to the state and action
    """

    def __init__(self, controller: Controller, getup_env: Any, handstand_env: Any):
        self._controller = controller
        self._src_env_action_scale = getup_env.env_cfg.action_scale
        self._tar_env_action_scale = handstand_env.env_cfg.action_scale

    def control(self, state: mjx_env.State, command: np.ndarray, data: mjx.Data) -> np.ndarray:
        """Control with state and action space adaptation."""
        # Adapt state for Getup control
        state = state[3:]  # remove first 3 linvel elements

        # Get control action
        action: np.ndarray = self._controller.control(state)

        # Adapt action space
        action = (self._src_env_action_scale * action - data.ctrl + data.qpos[7:]) / self._tar_env_action_scale

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

        if (self._active_type == Go1ControllerType.JOYSTICK) or (self._active_type == Go1ControllerType.GETUP):
            # Joystick controller requires command input
            return controller.control(state.obs["state"], self._command, state.data)

        # Other controllers use standard control
        return controller.control(state.obs["state"])


def create_go1_acrobat_controller_manager(
    controller_factory: ControllerFactory,
    params_builder: PPOParamsBuilder,
    controller_configs: Dict[Go1ControllerType, Dict[str, Any]],
    joystick_env: Go1Env,
    handstand_env: Go1Env,
    getup_env: Go1Env,
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
    controller_factory.register_controller(PPOParams, PPO)
    for controller_type, config in controller_configs.items():
        params = params_builder.build(config=config)
        base_controller = controller_factory.build(params=params)

        # Wrap joystick and getup controller with adapter and leave others as is
        # Regarding the adapter, refer to PPOEnvName2HandstandAdapter for more details
        if controller_type == Go1ControllerType.JOYSTICK:
            controllers[controller_type] = PPOJoystick2HandstandAdapter(
                controller=base_controller, joystick_env=joystick_env, handstand_env=handstand_env
            )
        elif controller_type == Go1ControllerType.GETUP:
            controllers[controller_type] = PPOGetup2HandstandAdapter(
                controller=base_controller, getup_env=getup_env, handstand_env=handstand_env
            )
        else:
            controllers[controller_type] = base_controller

    return Go1ControllerManager(controllers)
