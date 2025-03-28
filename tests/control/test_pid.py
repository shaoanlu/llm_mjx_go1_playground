import unittest
from typing import Any, Dict

from src.control.algorithms.pid import PID, PIDParams


class TestPID(unittest.TestCase):
    def setUp(self):
        pass

    def test_pid_params_dataclass(self):
        """Test PID parameters creation with different methods"""
        # Test direct creation
        params = PIDParams(kp=1.0, ki=0.1, kd=0.01)
        self.assertEqual(params.kp, 1.0)
        self.assertEqual(params.ki, 0.1)
        self.assertEqual(params.kd, 0.01)
        self.assertEqual(params.algorithm_type, "pid")

        # Test that parameters must be keyword arguments
        with self.assertRaises(TypeError):
            params = PIDParams(1.0, 0.1, 0.01)  # should fail as positional args

    def test_pid_params_building(self):
        """Test PID parameters builder"""
        config: Dict[str, Any] = {"algorithm_type": "pid", "kp": 1.0, "ki": 0.1, "kd": 0.01}
        params = PIDParams.from_dict(config)

        self.assertIsInstance(params, PIDParams)
        self.assertEqual(params.kp, 1.0)
        self.assertEqual(params.ki, 0.1)
        self.assertEqual(params.kd, 0.01)
        self.assertEqual(params.algorithm_type, "pid")

        # Test with missing required parameters
        invalid_config = {"algorithm_type": "pid", "kp": 1.0}  # missing ki and kd
        with self.assertRaises(Exception):
            PIDParams.from_dict(invalid_config)

    def test_pid_control(self):
        """Test first control step (when prev_error is None)"""
        # Given
        error = 1.0

        # When
        controller = PID(PIDParams(kp=1.0, ki=0.1, kd=0.01))
        controller.prev_error = 0.0
        output = controller.control(state=None, ref_state=None, error=error)

        # Then
        # - P term = kp * error = 1.0 * 1.0 = 1.0
        # - I term = ki * error * dt = 0.1 * 1.0 * 0.1 = 0.01
        # - D term = kd * (error - 0) / dt = 0.01 * 1.0 / 0.1 = 0.1
        expected_output = 1.0 + 0.01 + 0.1
        self.assertAlmostEqual(output, expected_output, places=6)


if __name__ == "__main__":
    unittest.main()
