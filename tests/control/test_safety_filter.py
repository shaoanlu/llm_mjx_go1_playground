import unittest
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_array_equal

from src.control.models import Simple2DRobot
from src.control.models.simple_robot import Simple2DRobotParams
from src.control.safety_filter import SafetyFilter, SafetyFilterParams
from src.control.state import Go1State
from src.control.algorithms.base import HighLevelCommand


class TestSafetyFilter(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = SafetyFilterParams(
            model=Simple2DRobot(Simple2DRobotParams(a=1.0, b=1.0)),  # robot shape as a circle of radius 1m
            max_output=(1.2, 0.5),
            min_output=(-1.2, -0.5),
            cbf_alpha=10,
            cbf_slack_penalty=10.0,
            cbf_kappa=0.5,
        )
        self.safety_filter = SafetyFilter(config=self.config)

        # Common test inputs
        self.state = Mock(spec=Go1State)
        self.state.position = np.array([0.0, 0.0, 0.0])  # Only XY coords used
        self.nominal_command = np.array([0.5, 0.3])

    def test_initialization(self):
        """Test proper initialization of SafetyFilter."""
        self.assertEqual(self.safety_filter.nx, 2)  # 2D control input
        self.assertEqual(self.safety_filter.nh, 1)  # Single composite barrier
        assert_array_equal(self.safety_filter.max_control, np.array([1.2, 0.5]))
        assert_array_equal(self.safety_filter.min_control, np.array([-1.2, -0.5]))

    def test_invalid_config(self):
        """Test that invalid configurations raise appropriate errors."""
        with self.assertRaises(AssertionError):
            SafetyFilterParams(cbf_alpha=-1.0)  # Invalid alpha

        with self.assertRaises(AssertionError):
            SafetyFilterParams(max_output=(1.0,), min_output=(-1.0, -0.5))  # Mismatched dimensions

        with self.assertRaises(AssertionError):
            SafetyFilterParams(max_output=(1.0, 0.5), min_output=(2.0, 1.0))  # Min greater than max

    def test_compute_command_no_obstacles(self):
        """Test behavior when no obstacles are present."""
        result = self.safety_filter.compute_command(
            state=self.state, command=self.nominal_command, obstacle_positions=[]
        )

        self.assertIsInstance(result, HighLevelCommand)
        assert_array_equal(result.value, self.nominal_command)
        self.assertIsNone(result.info)

    def test_compute_command_with_obstacles_and_zero_input(self):
        """Test safety filter behavior with obstacles present."""
        obstacles = [np.array([5.0, 0.0]), np.array([0.0, 5.0])]  # Obstacle ahead  # Obstacle to the right

        result = self.safety_filter.compute_command(
            state=self.state, command=self.nominal_command, obstacle_positions=obstacles
        )

        self.assertIsInstance(result, HighLevelCommand)
        self.assertIsNotNone(result.value)
        self.assertIsNotNone(result.info)

        # Verify command bounds
        self.assertTrue(
            np.all(result.value <= self.config.max_output), msg=f"{result.value=}, {self.config.max_output=}"
        )
        self.assertTrue(
            np.all(result.value >= self.config.min_output), msg=f"{result.value=}, {self.config.min_output=}"
        )

    def test_compute_command_with_obstacles_and_safe_input(self):
        """
        Test safety filter behavior with obstacles present.
        Input try to move away from the obstacle.
        """
        obstacles = [np.array([1.3, 0.0])]  # obstacle is 0.3m away from the robot
        nominal_command = np.array([-1.0, 0.0])  # Move away from obstacle

        result = self.safety_filter.compute_command(
            state=self.state, command=nominal_command, obstacle_positions=obstacles
        )

        # Verify command: expect CBF not modify the command
        np.testing.assert_allclose(
            result.value, nominal_command, rtol=1e-5, err_msg=f"{result.value=}, {nominal_command=}"
        )

    def test_compute_command_with_obstacles_and_adversarial_input(self):
        """
        Test safety filter behavior with obstacles present.
        Input try to move towards the obstacle.
        """
        obstacles = [np.array([0.7, 0.0])]  # obstacle collides with the robot
        nominal_command = np.array([1.0, 0.0])  # Move toward obstacle

        result = self.safety_filter.compute_command(
            state=self.state, command=nominal_command, obstacle_positions=obstacles
        )

        # Verify command: expect cbf to drive robot away from the obstacle with max negative command
        expected_safe_command = np.array([self.config.min_output[0], nominal_command[1]])
        np.testing.assert_allclose(
            result.value,
            expected_safe_command,
            rtol=1e-5,
            err_msg=f"{result.value=}, {expected_safe_command=}",
        )

    def test_composite_cbf_calculation(self):
        """Test the calculation of composite barrier function coefficients."""
        pos = np.array([0.0, 0.0])
        obstacles = np.array([[5.0, 0.0], [0.0, 5.0]])

        h, dhdx = self.safety_filter._calculate_composite_cbf_coeffs(pos, obstacles)

        # Check output shapes and types
        self.assertEqual(len(h), 1)  # Single composite barrier value
        self.assertEqual(len(dhdx), 1)  # Single row of derivatives
        self.assertEqual(len(dhdx[0]), 3)  # 2 state derivatives + 1 slack variable

        # Verify h is scalar and negative (unsafe) when too close to obstacles
        close_obstacles = np.array([[0.1, 0.1]])
        h_close, _ = self.safety_filter._calculate_composite_cbf_coeffs(pos, close_obstacles)
        self.assertLess(h_close[0], 0)

        # Verify h is positive (safe) when far from obstacles
        far_obstacles = np.array([[100.0, 100.0]])
        h_far, _ = self.safety_filter._calculate_composite_cbf_coeffs(pos, far_obstacles)
        self.assertGreater(h_far[0], 0)

    def test_input_validation(self):
        """Test input validation for compute_command."""
        invalid_state = Mock(spec=Go1State)
        invalid_state.position = np.array([0.0])  # Wrong dimension

        with self.assertRaises(AssertionError):
            self.safety_filter.compute_command(
                state=invalid_state, command=self.nominal_command, obstacle_positions=[np.array([5.0, 0.0])]
            )

        with self.assertRaises(AssertionError):
            self.safety_filter.compute_command(
                state=self.state,
                command=np.array([0.5]),  # Wrong control dimension
                obstacle_positions=[np.array([5.0, 0.0])],
            )

        with self.assertRaises(AssertionError):
            self.safety_filter.compute_command(
                state=self.state,
                command=self.nominal_command,
                obstacle_positions=[np.array([5.0])],  # Wrong obstacle dimension
            )

    @patch("src.control.safety_filter.CBFQPProblem")
    def test_qp_problem_setup(self, mock_qp_problem):
        """Test proper setup of the QP problem."""
        mock_qp_instance = Mock()
        mock_qp_problem.return_value = mock_qp_instance
        mock_qp_instance.solve.return_value = Mock(u=self.nominal_command)

        obstacles = [np.array([5.0, 0.0])]
        self.safety_filter.compute_command(
            state=self.state, command=self.nominal_command, obstacle_positions=obstacles
        )

        # Verify QP problem was created with correct dimensions
        mock_qp_problem.assert_called_once_with(nx=2, nh=1)

        # Verify create_matrices was called
        self.assertTrue(mock_qp_instance.create_matrices.called)

        # Verify solve was called
        self.assertTrue(mock_qp_instance.solve.called)


if __name__ == "__main__":
    unittest.main()
