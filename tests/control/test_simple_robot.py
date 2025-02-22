import unittest
from unittest.mock import patch

import numpy as np

# Import the modules to test - adjust import path as needed
from src.control.models.base import ControlAffineSystem, ControlAffineSystemParams
from src.control.models.simple_robot import Simple2DRobot, Simple2DRobotParams, _calculate_ellipse_closest_point


class TestSimple2DRobotParams(unittest.TestCase):
    """Tests for the Simple2DRobotParams dataclass"""

    def test_default_initialization(self):
        """Test that the config initializes with expected default values"""
        config = Simple2DRobotParams()
        self.assertEqual(config.a, 0.45)
        self.assertEqual(config.b, 0.3)
        self.assertEqual(config.x_dim, 2)
        self.assertEqual(config.u_dim, 2)
        self.assertIsInstance(config, ControlAffineSystemParams)

    def test_custom_initialization(self):
        """Test that the config can be initialized with custom values"""
        config = Simple2DRobotParams(a=10.0, b=5.0, x_dim=3, u_dim=1)
        self.assertEqual(config.a, 10.0)
        self.assertEqual(config.b, 5.0)
        self.assertEqual(config.x_dim, 3)
        self.assertEqual(config.u_dim, 1)


class TestCalculateEllipseClosestPoint(unittest.TestCase):
    """Tests for the _calculate_ellipse_closest_point helper function"""

    def test_ellipse_closest_point_calculation(self):
        """Test calculation of closest points on ellipse with various configurations"""

        test_params = [
            # center, a, b, points
            [(0, 0), 1, 1, np.array([[2, 0]])],  # Unit circle, point on x-axis
            [(0, 0), 2, 1, np.array([[4, 0]])],  # Ellipse, point on x-axis
            [(0, 0), 1, 2, np.array([[0, 4]])],  # Ellipse, point on y-axis
            [(1, 2), 2, 3, np.array([[5, 2]])],  # Ellipse with offset center, point on x-axis
            [(0, 0), 5, 3, np.array([[0, 0]])],  # Point at center
        ]
        for center, a, b, points in test_params:
            result = _calculate_ellipse_closest_point(center, a, b, points)

            # For each test point, verify the result is actually on the ellipse
            for i, point in enumerate(points):
                closest_point = result[i]
                center_array = np.array(center)

                # Calculate normalized distance from center to closest point
                dx = (closest_point[0] - center_array[0]) / a
                dy = (closest_point[1] - center_array[1]) / b
                dist_normalized = dx**2 + dy**2

                # The point should be on the ellipse (with small floating point tolerance)
                self.assertAlmostEqual(dist_normalized, 1.0, places=5)

                # Check that the closest point is actually closer than the original point
                dist_to_closest = np.linalg.norm(point - closest_point)
                dist_to_center = np.linalg.norm(point - center_array)

                # Skip this check if the point is at the center
                if not np.array_equal(point, center_array):
                    self.assertLessEqual(dist_to_closest, dist_to_center)

    def test_multiple_points(self):
        """Test handling multiple points at once"""
        center = (0, 0)
        a, b = 2, 1
        points = np.array([[4, 0], [0, 3], [2, 2], [-3, -1]])

        result = _calculate_ellipse_closest_point(center, a, b, points)

        # Check result shape
        self.assertEqual(result.shape, points.shape)

        # Verify all points are on the ellipse
        for i in range(len(points)):
            x, y = result[i]
            normalized_dist = (x / a) ** 2 + (y / b) ** 2
            self.assertAlmostEqual(normalized_dist, 1.0, places=5)

    def test_invalid_center_dimension(self):
        """Test that assertion error is raised when center has wrong dimension"""
        with self.assertRaises(AssertionError):
            _calculate_ellipse_closest_point((1, 2, 3), 1, 1, np.array([[0, 0]]))


class TestSimple2DRobot(unittest.TestCase):
    """Tests for the Simple2DRobot class"""

    def setUp(self):
        """Setup for each test with default configuration"""
        self.config = Simple2DRobotParams()
        self.robot = Simple2DRobot(config=self.config)

    def test_initialization(self):
        """Test robot initialization and inheritance"""
        self.assertIsInstance(self.robot, ControlAffineSystem)
        self.assertEqual(self.robot.config.a, 0.45)
        self.assertEqual(self.robot.config.b, 0.3)
        self.assertEqual(self.robot.config.x_dim, 2)
        self.assertEqual(self.robot.config.u_dim, 2)

    def test_f_x(self):
        """Test f_x method returns expected zero matrix"""
        x = np.array([1.0, 2.0])
        result = self.robot.f_x(x)

        expected = np.zeros((2, 2))
        np.testing.assert_array_equal(result, expected)

        # Test with different x_dim
        custom_config = Simple2DRobotParams(x_dim=3)
        custom_robot = Simple2DRobot(config=custom_config)
        result = custom_robot.f_x(np.array([1.0, 2.0, 3.0]))
        expected = np.zeros((3, 3))
        np.testing.assert_array_equal(result, expected)

    def test_g_x(self):
        """Test g_x method returns identity matrix of correct size"""
        x = np.array([3.0, 4.0])
        result = self.robot.g_x(x)

        expected = np.eye(2)
        np.testing.assert_array_equal(result, expected)

        # Test with different x_dim
        custom_config = Simple2DRobotParams(x_dim=4)
        custom_robot = Simple2DRobot(config=custom_config)
        result = custom_robot.g_x(np.array([1.0, 2.0, 3.0, 4.0]))
        expected = np.eye(4)
        np.testing.assert_array_equal(result, expected)

    @patch("src.control.models.simple_robot._calculate_ellipse_closest_point")
    def test_h(self, mock_calc_closest):
        """Test h method calculating barrier function value"""
        # Setup mock
        robot_pos = np.array([0, 0])
        obstacle_pos = np.array([[5, 5]])
        intersection_pt = np.array([[2, 2]])
        mock_calc_closest.return_value = intersection_pt

        # Distance from robot to obstacle = sqrt(50) = 7.07
        # Distance from robot to intersection = sqrt(8) = 2.83
        # Expected h = 50 - 8 = 42
        result = self.robot.h(robot_pos, obstacle_pos)

        # Verify the mock was called correctly
        mock_calc_closest.assert_called_once()
        np.testing.assert_array_equal(mock_calc_closest.call_args[1]["center"], robot_pos)
        self.assertEqual(mock_calc_closest.call_args[1]["a"], 0.45)
        self.assertEqual(mock_calc_closest.call_args[1]["b"], 0.3)
        np.testing.assert_array_equal(mock_calc_closest.call_args[1]["x"], obstacle_pos)

        # Check result
        expected = 50 - 8
        np.testing.assert_almost_equal(result, expected, decimal=5, err_msg=f"{result=}, {expected=}")

    def test_h_integration(self):
        """Integration test for h method with actual _calculate_ellipse_closest_point"""
        # Test with actual ellipse calculation (no mocking)
        robot_pos = np.array([0, 0])
        obstacle_pos = np.array([[1, 0]])  # Obstacle on x-axis

        result = self.robot.h(robot_pos, obstacle_pos)

        # Expected: ||obstacle - robot||^2 - ||ellipse_point - robot||^2
        # Ellipse point on x-axis would be (0.45, 0) since a=0.45
        expected = 1**2 - 0.45**2
        self.assertAlmostEqual(result, expected, places=5)

    def test_h_dot(self):
        """Test h_dot method calculating gradient of barrier function"""
        robot_pos = np.array([1, 2])
        obstacle_pos = np.array([[4, 6]])

        result = self.robot.h_dot(robot_pos, obstacle_pos)

        # Expected: 2 * (robot_pos - obstacle_pos)
        expected = 2 * (robot_pos - obstacle_pos)
        np.testing.assert_array_equal(result, expected)

    def test_h_with_multiple_obstacles(self):
        """Test h method with multiple obstacles"""
        robot_pos = np.array([0, 0])
        obstacles = np.array([[1, 0], [0, 1], [1, 1]])  # On x-axis  # On y-axis  # Diagonal

        result = self.robot.h(robot_pos, obstacles)

        # Check shape of result
        self.assertEqual(result.shape, (3,))

        # First obstacle (on x-axis)
        # Ellipse point would be at (0.45, 0)
        expected1 = 1**2 - 0.45**2

        # Second obstacle (on y-axis)
        # Ellipse point would be at (0, 0.3)
        expected2 = 1**2 - 0.3**2

        # Third obstacle (diagonal)
        # Distance calculation is more complex, but we can verify it's positive

        self.assertAlmostEqual(result[0], expected1, places=5)
        self.assertAlmostEqual(result[1], expected2, places=5)
        self.assertTrue(result[2] > 0)  # Barrier function should be positive when not in collision


class TestSimple2DRobotEdgeCases(unittest.TestCase):
    """Tests for edge cases and special conditions"""

    def setUp(self):
        self.config = Simple2DRobotParams()
        self.robot = Simple2DRobot(config=self.config)

    def test_zero_sized_ellipse(self):
        """Test behavior with a zero-sized ellipse (a=b=0)"""
        zero_config = Simple2DRobotParams(a=0.0, b=0.0)
        zero_robot = Simple2DRobot(config=zero_config)

        robot_pos = np.array([0, 0])
        obstacle_pos = np.array([[5, 5]])

        with self.assertRaises(Exception):
            # This should raise some exception since dividing by zero in calculations
            zero_robot.h(robot_pos, obstacle_pos)

    def test_identical_positions(self):
        """Test when robot and obstacle are at the same position"""
        robot_pos = np.array([10, 10])
        obstacle_pos = np.array([[10, 10]])

        # The h value should be negative (collision)
        result = self.robot.h(robot_pos, obstacle_pos)
        self.assertTrue((result < 0).all())

        # h_dot should be zero vector as robot and obstacle are at the same position
        h_dot_result = self.robot.h_dot(robot_pos, obstacle_pos)
        np.testing.assert_array_equal(h_dot_result, np.zeros_like(obstacle_pos))

    def test_circular_robot(self):
        """Test with a circular robot (a=b)"""
        circle_config = Simple2DRobotParams(a=10.0, b=10.0)  # a circle
        circle_robot = Simple2DRobot(config=circle_config)

        robot_pos = np.array([0, 0])

        # Test points at same distance but different angles
        obstacles = np.array(
            [
                [20, 0],  # Right
                [0, 20],  # Up
                [14.14, 14.14],  # Diagonal (approximately same distance)
            ]
        )

        result = circle_robot.h(robot_pos, obstacles)

        # For a circle, the distance from center to collision point should be same in all directions
        # So the h values should be approximately equal
        self.assertAlmostEqual(result[0], result[1], places=0)  # place set as 0 as the unit is pixel
        self.assertAlmostEqual(result[0], result[2], places=0)


if __name__ == "__main__":
    unittest.main()
