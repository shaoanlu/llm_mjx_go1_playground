import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, Mock, patch

import jax.numpy as jnp
import numpy as np

from src.control.algorithms.base import Controller, ControllerParams
from src.control.algorithms.polar_coord_control import PolarCoordinateControllerParams
from src.control.algorithms.seq_pos_control import SequentialControllerParams
from src.control.controller_factory import ControllerFactory
from src.control.position_controller import (
    PositionCommandInfo,
    PositionController,
    PositionControllerParams,
)
from src.utils import load_yaml


class TestControllerParams(unittest.TestCase):
    def test_position_controller_params(self):
        """Test PositionControllerParams initialization and defaults"""
        params = PositionControllerParams()
        self.assertEqual(params.command_dim, 3)
        self.assertEqual(params.arrival_threshold, 0.1)
        self.assertEqual(params.max_linear_velocity, 1.5)
        self.assertEqual(params.max_angular_velocity, np.pi / 2)
        self.assertIsInstance(params.primary_controller, SequentialControllerParams)
        self.assertIsInstance(params.fallback_controller, SequentialControllerParams)

    def test_position_controller_params_from_dict(self):
        """Test PositionControllerParams initialization from dictionary"""
        # Test with custom values and nested controllers
        config_dict = load_yaml("tests/control/fixtures/position_controller.yaml")

        params = PositionControllerParams.from_dict(config_dict)

        # Verify all parameters were set correctly
        self.assertEqual(params.command_dim, 1)
        self.assertEqual(params.arrival_threshold, 1.2)
        self.assertEqual(params.max_linear_velocity, 3.4)
        self.assertEqual(params.max_angular_velocity, 5.6)
        self.assertEqual(params.algorithm_type, "position_controller")

        # Verify nested controllers were instantiated correctly
        self.assertIsInstance(params.primary_controller, SequentialControllerParams)
        self.assertEqual(params.primary_controller.yaw_control_threshold, 0.123)
        self.assertEqual(params.primary_controller.yaw_control_gain, 0.456)
        self.assertEqual(params.primary_controller.linear_control_gain, 0.789)
        self.assertEqual(params.primary_controller.algorithm_type, "seq_controller")

        self.assertIsInstance(params.fallback_controller, PolarCoordinateControllerParams)
        self.assertEqual(params.fallback_controller.linear_control_gain, 123.0)
        self.assertEqual(params.fallback_controller.yaw_control_gain1, 456.0)
        self.assertEqual(params.fallback_controller.yaw_control_gain2, 789.0)
        self.assertEqual(params.fallback_controller.rho_threshold, 10.0)
        self.assertEqual(params.fallback_controller.algorithm_type, "polar_coord_controller")

        # Test that original dict was not modified
        original_dict = load_yaml("tests/control/fixtures/position_controller.yaml")
        params = PositionControllerParams.from_dict(original_dict)
        self.assertEqual(original_dict["primary_controller"]["algorithm_type"], "seq_controller")
        self.assertEqual(original_dict["fallback_controller"]["algorithm_type"], "polar_coord_controller")


class TestPositionCommandInfo(unittest.TestCase):
    """Test PositionCommandInfo dataclass"""

    def test_initialization_and_immutability(self):
        """Test initialization and immutability of PositionCommandInfo"""
        pos = np.array([1.0, 2.0, 3.0])
        target_pos = np.array([4.0, 5.0, 6.0])
        info = PositionCommandInfo(pos=pos, target_pos=target_pos, is_arrived=False)

        self.assertTrue(np.array_equal(info.pos, pos))
        self.assertTrue(np.array_equal(info.target_pos, target_pos))
        self.assertFalse(info.is_arrived)

        # Test immutability
        with self.assertRaises(FrozenInstanceError):
            info.is_arrived = True


class TestPositionController(unittest.TestCase):
    """Test PositionController behavior"""

    def setUp(self):
        self.factory = Mock(spec=ControllerFactory)
        self.config = PositionControllerParams(
            arrival_threshold=0.1,
            max_linear_velocity=1.5,
            max_angular_velocity=np.pi / 2,
        )
        self.primary_controller = Mock(spec=Controller)
        self.fallback_controller = Mock(spec=Controller)

        # Setup mock factory
        def mock_build(params):
            if issubclass(type(params), ControllerParams):
                if params is self.config.primary_controller:
                    return self.primary_controller
                return self.fallback_controller
            return None

        self.factory.build = mock_build
        self.controller = PositionController(self.factory, self.config)

    def test_initialization(self):
        """Test controller initialization"""
        self.assertIsNotNone(self.controller._controllers.get("primary_controller"))
        self.assertIsNotNone(self.controller._controllers.get("fallback_controller"))
        self.assertIsNone(self.controller.prev_command)

    def test_compute_command_arrival(self):
        """Test command computation when robot has arrived at target"""
        state = MagicMock()
        state.data.qpos = np.zeros(19)  # 7 base DOFs + 12 joint DOFs
        state.data.site_xmat = np.stack([np.eye(3)] * 6, axis=0)
        state.data.site_xpos = np.zeros((6, 3))
        target_position = np.array([0.05, 0.05])  # Within arrival threshold

        result = self.controller.compute_command(state, target_position)
        np.testing.assert_array_equal(result.command, jnp.zeros(3), err_msg=f"{result.command=}")
        self.assertTrue(result.info.is_arrived)

    @patch.object(PositionController, "_primary_control")
    def test_compute_command_primary_control(self, mock_primary):
        """Test command computation using primary controller"""
        state = MagicMock()
        state.data.qpos = np.zeros(19)
        state.data.site_xmat = np.stack([np.eye(3)] * 6, axis=0)
        state.data.site_xpos = np.zeros((6, 3))
        target_position = np.array([1.0, 1.0])  # Far from current position

        mock_primary.return_value = np.array([1.0, 0.0, 0.5])
        result = self.controller.compute_command(state, target_position)

        self.assertTrue(mock_primary.called)
        np.testing.assert_array_almost_equal(result.command, jnp.array([1.0, 0.0, 0.5]), err_msg=f"{result.command=}")

    @patch.object(PositionController, "_primary_control", side_effect=Exception)
    @patch.object(PositionController, "_fallback_control")
    def test_compute_command_fallback(self, mock_fallback, mock_primary):
        """Test fallback to secondary controller when primary fails"""
        state = MagicMock()
        state.data.qpos = np.zeros(19)
        state.data.site_xmat = np.stack([np.eye(3)] * 6, axis=0)
        state.data.site_xpos = np.zeros((6, 3))
        target_position = np.array([1.0, 1.0])

        mock_fallback.return_value = np.array([0.5, 0.0, 0.3])
        result = self.controller.compute_command(state, target_position)

        self.assertTrue(mock_primary.called)
        self.assertTrue(mock_fallback.called)
        np.testing.assert_array_almost_equal(result.command, jnp.array([0.5, 0.0, 0.3]), err_msg=f"{result.command=}")

    def test_post_process_command(self):
        """Test command post-processing (clipping)"""
        command = np.array([2.0, 0.0, 2.0])  # Exceeds max velocities
        processed = self.controller._post_process_command(command)

        np.testing.assert_array_almost_equal(
            processed,
            jnp.array([1.5, 0.0, np.pi / 2]),  # Should be clipped to max values
        )


if __name__ == "__main__":
    unittest.main()
