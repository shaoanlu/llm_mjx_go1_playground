import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import jax.numpy as jnp

from src.control.algorithms.base import Controller
from src.control.algorithms.mlp import MLPPolicy, MLPPolicyParams
from src.control.acrobat_controller import (
    Go1ControllerType,
    Go1ControllerManager,
    MLPPolicyGetup2HandstandAdapter,
    MLPPolicyJoystick2HandstandAdapter,
    JOYSTICK_ENV_ACTION_SCALE,
    HANDSTAND_ENV_ACTION_SCALE,
    GETUP_ENV_ACTION_SCALE,
    JOYSTICK_ENV_DEFAULT_POSE,
    create_acrobat_controller_manager,
)


class TestGo1ControllerManager(unittest.TestCase):
    def setUp(self):
        # Create mock controllers for each type
        self.mock_footstand = Mock(spec=Controller)
        self.mock_joystick = Mock(spec=Controller)
        self.mock_handstand = Mock(spec=Controller)
        self.mock_getup = Mock(spec=Controller)

        self.controllers = {
            Go1ControllerType.FOOTSTAND: self.mock_footstand,
            Go1ControllerType.JOYSTICK: self.mock_joystick,
            Go1ControllerType.HANDSTAND: self.mock_handstand,
            Go1ControllerType.GETUP: self.mock_getup,
        }

        self.manager = Go1ControllerManager(self.controllers)

    def test_initial_state(self):
        """Test initial state of controller manager"""
        self.assertEqual(self.manager._active_type, Go1ControllerType.FOOTSTAND)
        np.testing.assert_array_equal(
            self.manager._command, jnp.zeros(3), err_msg=f"{self.manager._command=}"
        )

    def test_set_command(self):
        """Test setting command with valid and invalid inputs"""
        valid_command = np.array([1.0, -0.5, 0.2])
        self.manager.set_command(valid_command)
        np.testing.assert_array_equal(
            self.manager._command, valid_command, err_msg=f"{self.manager._command=}"
        )

        invalid_command = np.array([1.0, -0.5])  # invalid shape
        with self.assertRaises(ValueError):
            self.manager.set_command(invalid_command)

    def test_switch_controller(self):
        """Test switching between controllers"""
        self.manager.switch_controller(Go1ControllerType.JOYSTICK)
        self.assertEqual(self.manager._active_type, Go1ControllerType.JOYSTICK)

        with self.assertRaises(ValueError):
            self.manager.switch_controller("invalid_controller")

    def test_control_delegation(self):
        """Test if control calls are properly delegated to active controller"""
        mock_state = MagicMock()
        mock_state.obs = {"state": np.zeros(45)}
        mock_state.data = MagicMock()

        # Test with different controller calls
        self.manager.switch_controller(Go1ControllerType.FOOTSTAND)
        self.manager.control(mock_state)
        self.mock_footstand.control.assert_called_once()

        self.manager.switch_controller(Go1ControllerType.JOYSTICK)
        self.manager.control(mock_state)
        self.mock_joystick.control.assert_called_once()

        self.manager.switch_controller(Go1ControllerType.HANDSTAND)
        self.manager.control(mock_state)
        self.mock_handstand.control.assert_called_once()

        self.manager.switch_controller(Go1ControllerType.GETUP)
        self.manager.control(mock_state)
        self.mock_getup.control.assert_called_once()


class TestMLPPolicyJoystick2HandstandAdapter(unittest.TestCase):
    def setUp(self):
        self.mock_adaptee_controller = Mock(spec=Controller)
        self.adapter = MLPPolicyJoystick2HandstandAdapter(self.mock_adaptee_controller)

    def test_control_adaptation(self):
        """Test state and action space adaptation"""
        # Mock input state and data
        state = jnp.array([1.0, 2.0, 3.0])
        command = jnp.array([0.1, 0.2, 0.3])
        mjx_state_data = MagicMock()
        mjx_state_data.ctrl = jnp.zeros(12)

        # Mock controller response
        mock_action = jnp.ones(12)
        self.mock_adaptee_controller.control.return_value = mock_action

        # Call adapter
        result = self.adapter.control(state, command, mjx_state_data)

        # Verify state adaptation
        expected_adapted_state = jnp.concatenate([state, command])
        self.mock_adaptee_controller.control.assert_called_once()
        np.testing.assert_array_equal(
            self.mock_adaptee_controller.control.call_args[0][0],
            expected_adapted_state,
            err_msg=f"{self.mock_adaptee_controller.control.call_args=}",
        )

        # Verify action adaptation
        expected_action = (
            JOYSTICK_ENV_ACTION_SCALE * mock_action
            - mjx_state_data.ctrl
            + JOYSTICK_ENV_DEFAULT_POSE
        ) / HANDSTAND_ENV_ACTION_SCALE
        np.testing.assert_array_equal(result, expected_action, err_msg=f"{result=}")


class TestMLPPolicyGetup2HandstandAdapter(unittest.TestCase):
    def setUp(self):
        self.mock_adaptee_controller = Mock(spec=Controller)
        self.adapter = MLPPolicyGetup2HandstandAdapter(self.mock_adaptee_controller)

    def test_control_adaptation(self):
        """Test state and action space adaptation"""
        # Mock input state and data
        state = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # First 3 elements are linvel
        mjx_state_data = MagicMock()
        mjx_state_data.ctrl = jnp.zeros(12)
        mjx_state_data.qpos = jnp.concatenate(
            [jnp.zeros(7), jnp.ones(12)]
        )  # 7 base DOFs + 12 actuated DOFs
        command = jnp.zeros(3)  # Unused in this adapter

        # Mock controller response
        mock_action = jnp.ones(12)
        self.mock_adaptee_controller.control.return_value = mock_action

        # Call adapter
        result = self.adapter.control(state, command, mjx_state_data)

        # Verify state adaptation
        expected_adapted_state = state[3:]  # Remove first 3 linvel elements
        self.mock_adaptee_controller.control.assert_called_once()
        np.testing.assert_array_equal(
            self.mock_adaptee_controller.control.call_args[0][0],
            expected_adapted_state,
            err_msg=f"{self.mock_adaptee_controller.control.call_args=}",
        )

        # Verify action adaptation
        expected_action = (
            GETUP_ENV_ACTION_SCALE * mock_action - mjx_state_data.ctrl + mjx_state_data.qpos[7:]
        ) / HANDSTAND_ENV_ACTION_SCALE
        np.testing.assert_array_equal(result, expected_action, err_msg=f"{result=}")


@patch("src.control.controller_factory.ControllerFactory")
@patch("src.control.controller_factory.ConfigFactory")
class TestCreateAcrobatControllerManager(unittest.TestCase):
    def setUp(self):
        self.controller_configs = {
            Go1ControllerType.JOYSTICK: {"type": "mlp", "params": {}},
            Go1ControllerType.HANDSTAND: {"type": "mlp", "params": {}},
            Go1ControllerType.FOOTSTAND: {"type": "mlp", "params": {}},
            Go1ControllerType.GETUP: {"type": "mlp", "params": {}},
        }

    def test_controller_creation(self, mock_config_factory, mock_controller_factory):
        """Test creation of controller manager with all controller types"""
        # Setup mock returns
        mock_params = MagicMock()
        mock_config_factory.build.return_value = mock_params

        mock_controller = Mock(spec=Controller)
        mock_controller_factory.build.return_value = mock_controller

        # Create controller manager
        manager = create_acrobat_controller_manager(
            mock_controller_factory, mock_config_factory, self.controller_configs
        )

        # Verify registrations
        mock_config_factory.register_config.assert_called_once_with("mlp", MLPPolicyParams)
        mock_controller_factory.register_controller.assert_called_once_with(
            MLPPolicyParams, MLPPolicy
        )

        # Verify controller creation for each type
        self.assertEqual(mock_controller_factory.build.call_count, len(Go1ControllerType))
        self.assertEqual(mock_config_factory.build.call_count, len(Go1ControllerType))

        # Verify adapter wrapping for specific controllers
        self.assertIsInstance(
            manager._controllers[Go1ControllerType.JOYSTICK],
            MLPPolicyJoystick2HandstandAdapter,
            msg=f"{type( manager._controllers[Go1ControllerType.JOYSTICK])=}",
        )
        self.assertIsInstance(
            manager._controllers[Go1ControllerType.GETUP],
            MLPPolicyGetup2HandstandAdapter,
            msg=f"{type( manager._controllers[Go1ControllerType.GETUP])=}",
        )


if __name__ == "__main__":
    unittest.main()
