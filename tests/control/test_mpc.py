import unittest

import numpy as np

from src.control.algorithm.mpc import MPC, MPCParams, MPCParamsBuilder


class TestMPC(unittest.TestCase):
    def setUp(self):
        pass

    def test_mpc_params_dataclass(self):
        """Test MPC parameters creation with different methods"""
        # Test direct creation
        params = MPCParams(Q=np.eye(2), R=np.eye(1))
        self.assertTrue(np.array_equal(params.Q, np.eye(2)))
        self.assertTrue(np.array_equal(params.R, np.eye(1)))
        self.assertEqual(params.algorithm_type, "mpc")

        # Test that parameters must be keyword arguments
        with self.assertRaises(TypeError):
            params = MPCParams(np.eye(5), np.eye(4))  # should fail as positional args

    def test_mpc_params_builder(self):
        """Test MPC parameters builder"""
        config = {"algorithm_type": "mpc", "Q": np.eye(2), "R": np.eye(1)}
        builder = MPCParamsBuilder()
        params = builder.build(config)
        self.assertIsInstance(params, MPCParams)
        self.assertTrue(np.array_equal(params.Q, np.eye(2)))
        self.assertTrue(np.array_equal(params.R, np.eye(1)))
        self.assertEqual(params.algorithm_type, "mpc")

        # Test with missing required parameters
        invalid_config = {"algorithm_type": "mpc", "Q": np.eye(2)}  # missing R parameter
        with self.assertRaises(Exception):
            builder.build(invalid_config)

    def test_mpc_instantiation(self):
        controller = MPC(MPCParams(Q=np.eye(2), R=np.eye(1)))
        self.assertIsInstance(controller, MPC)


if __name__ == "__main__":
    unittest.main()
