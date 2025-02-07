import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, Mock, patch

import jax.numpy as jnp
import numpy as np

from src.control.algorithms.base import Controller, ControllerParams
from src.control.controller_factory import ControllerFactory
from src.control.position_controller import (
    PolarCoordinateController,
    PolarCoordinateControllerParams,
    PositionCommandInfo,
    PositionController,
    PositionControllerParams,
    SequentialController,
    SequentialControllerParams,
    create_position_controller,
)
from src.control.state import Go1State


class TestControllerParams(unittest.TestCase):
    """Test parameter dataclasses"""

    def test_sequential_controller_params(self):
        """Test SequentialControllerParams initialization and defaults"""
        params = SequentialControllerParams()
        self.assertEqual(params.yaw_control_threshold, np.pi / 18)
        self.assertEqual(params.yaw_control_gain, 7.0)
        self.assertEqual(params.linear_control_gain, 2.0)
        self.assertEqual(params.algorithm_type, "seq_controller")

        # Test custom values
        custom_params = SequentialControllerParams(
            yaw_control_threshold=0.5, yaw_control_gain=5.0, linear_control_gain=3.0
        )
        self.assertEqual(custom_params.yaw_control_threshold, 0.5)
        self.assertEqual(custom_params.yaw_control_gain, 5.0)
        self.assertEqual(custom_params.linear_control_gain, 3.0)

    def test_polar_coordinate_controller_params(self):
        """Test PolarCoordinateControllerParams initialization and defaults"""
        params = PolarCoordinateControllerParams()
        self.assertEqual(params.linear_control_gain, 2.0)
        self.assertEqual(params.yaw_control_gain1, 7.0)
        self.assertEqual(params.yaw_control_gain2, 1.0)
        self.assertEqual(params.rho_threshold, 0.1)
        self.assertEqual(params.algorithm_type, "polar_coord_controller")

    def test_position_controller_params(self):
        """Test PositionControllerParams initialization and defaults"""
        params = PositionControllerParams()
        self.assertEqual(params.command_dim, 3)
        self.assertEqual(params.arrival_threshold, 0.1)
        self.assertEqual(params.max_linear_velocity, 1.5)
        self.assertEqual(params.max_angular_velocity, np.pi / 2)
        self.assertIsInstance(params.primary_controller, SequentialControllerParams)
        self.assertIsInstance(params.fallback_controller, SequentialControllerParams)


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


class TestSequentialController(unittest.TestCase):
    """Test SequentialController behavior"""

    def setUp(self):
        self.config = SequentialControllerParams()
        self.controller = SequentialController(self.config)

    def test_yaw_control(self):
        """Test yaw control when yaw error is above threshold"""
        state = Go1State(position=np.array([0, 0, 0]), yaw=0)
        ref_state = Go1State(position=np.array([1, 1, 0]), yaw=np.pi / 2)

        control = self.controller.control(state, ref_state)
        expected_control = np.array([0.0, 0.0, self.config.yaw_control_gain * np.pi / 2])
        np.testing.assert_array_almost_equal(control, expected_control, err_msg=f"{control=}")

    def test_linear_control(self):
        """Test linear control when yaw error is below threshold"""
        state = Go1State(position=np.array([0, 0, 0]), yaw=0)
        ref_state = Go1State(position=np.array([1, 0, 0]), yaw=0)

        control = self.controller.control(state, ref_state)
        expected_control = np.array([self.config.linear_control_gain * 1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(control, expected_control, err_msg=f"{control=}")


class TestPolarCoordinateController(unittest.TestCase):
    """Test PolarCoordinateController behavior"""

    def setUp(self):
        self.config = PolarCoordinateControllerParams()
        self.controller = PolarCoordinateController(self.config)

    def test_far_control(self):
        """Test control when robot is far from target (rho > threshold)"""
        state = Go1State(position=np.array([0, 0, 0]), yaw=0)
        ref_state = Go1State(position=np.array([1, 1, 0]), yaw=0)

        control = self.controller.control(state, ref_state)
        self.assertEqual(len(control), 3)
        self.assertGreater(control[0], 0)  # Should move forward

    def test_near_control(self):
        """Test control when robot is near target (rho < threshold)"""
        state = Go1State(position=np.array([0, 0, 0]), yaw=0)
        ref_state = Go1State(position=np.array([0.05, 0, 0]), yaw=np.pi / 4)

        control = self.controller.control(state, ref_state)
        self.assertEqual(len(control), 3, msg=f"{len(control)=}")
        self.assertNotEqual(control[2], 0, msg=f"{control=}")  # Should rotate to target yaw


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
        np.testing.assert_array_almost_equal(
            result.command, jnp.array([1.0, 0.0, 0.5]), err_msg=f"{result.command=}"
        )

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
        np.testing.assert_array_almost_equal(
            result.command, jnp.array([0.5, 0.0, 0.3]), err_msg=f"{result.command=}"
        )

    def test_post_process_command(self):
        """Test command post-processing (clipping)"""
        command = np.array([2.0, 0.0, 2.0])  # Exceeds max velocities
        processed = self.controller._post_process_command(command)

        np.testing.assert_array_almost_equal(
            processed, jnp.array([1.5, 0.0, np.pi / 2])  # Should be clipped to max values
        )


class TestCreatePositionController(unittest.TestCase):
    """Test position controller factory function"""

    def test_controller_creation(self):
        """Test creation of position controller with factory"""
        factory = Mock(spec=ControllerFactory)
        config = PositionControllerParams()

        controller = create_position_controller(factory, config)

        # Verify controller registrations
        factory.register_controller.assert_any_call(
            SequentialControllerParams, SequentialController
        )
        factory.register_controller.assert_any_call(
            PolarCoordinateControllerParams, PolarCoordinateController
        )

        self.assertIsInstance(controller, PositionController, msg=f"{type(controller)=}")


if __name__ == "__main__":
    unittest.main()
