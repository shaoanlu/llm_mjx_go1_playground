import unittest

import numpy as np

from src.control.algorithms.seq_pos_control import SequentialController, SequentialControllerParams
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


class TestSequentialController(unittest.TestCase):
    """Test SequentialController behavior"""

    def setUp(self):
        self.config = SequentialControllerParams()
        self.controller = SequentialController(self.config)

    def test_yaw_control(self):
        """Test yaw control when yaw error is above threshold"""
        state = Go1State(position=np.array([0, 0, 0]), yaw=0)

        # Case 1: large positive yaw error
        ref_state = Go1State(position=np.array([1, 1, 0]), yaw=np.pi / 2)
        control = self.controller.control(state, ref_state)
        expected_control = np.array([0.0, 0.0, self.config.yaw_control_gain * np.pi / 2])
        np.testing.assert_array_almost_equal(control, expected_control, err_msg=f"{control=}")

        # Case 2: large negative yaw error
        ref_state = Go1State(position=np.array([1, 1, 0]), yaw=-np.pi / 3)
        control = self.controller.control(state, ref_state)
        expected_control = np.array([0.0, 0.0, self.config.yaw_control_gain * -np.pi / 3])
        np.testing.assert_array_almost_equal(control, expected_control, err_msg=f"{control=}")

    def test_linear_control(self):
        """Test linear control when yaw error is below threshold"""
        state = Go1State(position=np.array([0, 0, 0]), yaw=0)
        ref_state = Go1State(position=np.array([1, 0, 0]), yaw=0)

        control = self.controller.control(state, ref_state)
        expected_control = np.array([self.config.linear_control_gain * 1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(control, expected_control, err_msg=f"{control=}")


if __name__ == "__main__":
    unittest.main()
