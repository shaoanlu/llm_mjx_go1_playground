import unittest

import numpy as np

from src.control.algorithms.polar_coord_control import PolarCoordinateController, PolarCoordinateControllerParams
from src.control.state import Go1State


class TestControllerParams(unittest.TestCase):
    def test_polar_coordinate_controller_params(self):
        """Test PolarCoordinateControllerParams initialization and defaults"""
        params = PolarCoordinateControllerParams()
        self.assertEqual(params.linear_control_gain, 2.0)
        self.assertEqual(params.yaw_control_gain1, 7.0)
        self.assertEqual(params.yaw_control_gain2, 1.0)
        self.assertEqual(params.rho_threshold, 0.1)
        self.assertEqual(params.algorithm_type, "polar_coord_controller")


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
