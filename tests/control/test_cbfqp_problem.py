import unittest

import numpy as np

from src.control.algorithms.cbfqp_problem import CBFQPProblem


class TestCBFQPProblem(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Common test dimensions
        self.nx = 2  # Control dimensions (e.g., x and y)
        self.nh = 1  # Number of barrier functions

        # Create a CBFQPProblem instance
        self.qp_problem = CBFQPProblem(nx=self.nx, nh=self.nh)

        # Common test parameters
        self.h = [1.0]  # Single barrier function value h(x)
        self.coeffs_dhdx = [[1.0, 0.0] + [1.0]]  # Lgh(x) and slack vaariable coefficients
        self.nominal_control = np.array([0.5, 0.5])
        self.max_control = np.array([1.0, 1.0])
        self.min_control = np.array([-1.0, -1.0])
        self.cbf_alpha = 1.0
        self.slack_penalty = 1e3
        self.disturbance_h_dot = [0.1]

    def test_initialization(self):
        """Test proper initialization of CBFQPProblem."""
        self.assertEqual(self.qp_problem.nx, self.nx)
        self.assertEqual(self.qp_problem.nh, self.nh)

    def test_create_matrices_dimensions(self):
        """Test if created matrices have correct dimensions."""
        qp_data = self.qp_problem.create_matrices(
            h=self.h,
            coeffs_dhdx=self.coeffs_dhdx,
            nominal_control=self.nominal_control,
            max_control=self.max_control,
            min_control=self.min_control,
            cbf_alpha=self.cbf_alpha,
            slack_penalty=self.slack_penalty,
            disturbance_h_dot=self.disturbance_h_dot,
        )

        # Check dimensions
        self.assertEqual(qp_data.P.shape, (self.nx + self.nh, self.nx + self.nh))
        self.assertEqual(qp_data.q.shape, (self.nx + self.nh,))
        self.assertEqual(qp_data.A.shape, (self.nx + self.nh + self.nh, self.nx + self.nh))
        self.assertEqual(qp_data.l.shape, (self.nx + self.nh + self.nh,))
        self.assertEqual(qp_data.u.shape, (self.nx + self.nh + self.nh,))

    def test_cost_matrix_structure(self):
        """Test if cost matrix P has correct structure."""
        qp_data = self.qp_problem.create_matrices(
            h=self.h,
            coeffs_dhdx=self.coeffs_dhdx,
            nominal_control=self.nominal_control,
            max_control=self.max_control,
            min_control=self.min_control,
            cbf_alpha=self.cbf_alpha,
            slack_penalty=self.slack_penalty,
        )

        # Check identity structure for control variables
        np.testing.assert_array_almost_equal(qp_data.P[: self.nx, : self.nx], np.eye(self.nx))

        # Check slack penalty
        np.testing.assert_array_almost_equal(qp_data.P[self.nx :, self.nx :], np.eye(self.nh) * self.slack_penalty)

    def test_feasible_problem(self):
        """Test if a feasible problem returns valid solution."""
        qp_data = self.qp_problem.create_matrices(
            h=self.h,
            coeffs_dhdx=self.coeffs_dhdx,
            nominal_control=self.nominal_control,
            max_control=self.max_control,
            min_control=self.min_control,
            cbf_alpha=self.cbf_alpha,
            slack_penalty=self.slack_penalty,
        )

        solution = self.qp_problem.solve(qp_data)

        # Check if solution exists
        self.assertIsNotNone(solution.u)
        self.assertIsNotNone(solution.slack)

        # Check dimensions
        self.assertEqual(len(solution.u), self.nx)
        self.assertEqual(len(solution.slack), self.nh)

        # Check if solution respects bounds
        np.testing.assert_array_less(solution.u, self.max_control + 1e-10)
        np.testing.assert_array_less(self.min_control - 1e-10, solution.u)

    def test_cbf_constraint_satisfaction(self):
        """Test if CBF constraints are satisfied in the solution."""
        qp_data = self.qp_problem.create_matrices(
            h=self.h,
            coeffs_dhdx=self.coeffs_dhdx,
            nominal_control=self.nominal_control,
            max_control=self.max_control,
            min_control=self.min_control,
            cbf_alpha=self.cbf_alpha,
            slack_penalty=self.slack_penalty,
        )

        solution = self.qp_problem.solve(qp_data)

        # Check CBF constraint: h_dot >= -alpha * h - slack
        for i in range(self.nh):
            h_dot = np.dot(self.coeffs_dhdx[i][self.nx], solution.u)  # Lgh(x) * u
            cbf_constraint = h_dot + self.cbf_alpha * self.h[i] + solution.slack[i]
            self.assertTrue((cbf_constraint >= -1e-10).all(), msg=f"{cbf_constraint=}")

    def test_default_disturbance(self):
        """Test if default disturbance handling works correctly."""
        # Create matrices without specifying disturbance
        qp_data = self.qp_problem.create_matrices(
            h=self.h,
            coeffs_dhdx=self.coeffs_dhdx,
            nominal_control=self.nominal_control,
            max_control=self.max_control,
            min_control=self.min_control,
            cbf_alpha=self.cbf_alpha,
            slack_penalty=self.slack_penalty,
        )

        # Check if default disturbance is zero
        expected_l = np.concatenate(
            [[-self.cbf_alpha * h_ for h_ in self.h], self.min_control, np.full(self.nh, -np.inf)]
        )
        np.testing.assert_array_almost_equal(qp_data.l[: self.nh], expected_l[: self.nh])

    def test_high_disturbance_response(self):
        """Test system response to high disturbance values."""
        high_disturbance = [1.0]  # Significant disturbance

        qp_data = self.qp_problem.create_matrices(
            h=self.h,
            coeffs_dhdx=self.coeffs_dhdx,
            nominal_control=self.nominal_control,
            max_control=self.max_control,
            min_control=self.min_control,
            cbf_alpha=self.cbf_alpha,
            slack_penalty=self.slack_penalty,
            disturbance_h_dot=high_disturbance,
        )

        solution = self.qp_problem.solve(qp_data)

        # Solution should still exist and respect bounds
        self.assertIsNotNone(solution.u)
        self.assertIsNotNone(solution.slack)
        np.testing.assert_array_less(solution.u, self.max_control + 1e-10)
        np.testing.assert_array_less(self.min_control - 1e-10, solution.u)

    def test_multiple_barrier_functions(self):
        """Test handling of multiple barrier functions."""
        # Create a new problem with multiple barrier functions
        nx, nh = 2, 2
        multi_cbf_problem = CBFQPProblem(nx=nx, nh=nh)

        # Test data for multiple barriers
        h = [1.0, 0.5]
        dhdx = [[1.0, 0.0], [0.0, 1.0]]  # simple gradient for h
        coeffs_slack = [[1.0, 0.0], [0.0, 1.0]]  # one slack variable per barrier function
        coeffs_dhdx = np.hstack([dhdx, coeffs_slack])  # equals to [x + y for x, y in zip(dhdx, coeffs_slack)]

        qp_data = multi_cbf_problem.create_matrices(
            h=h,
            coeffs_dhdx=coeffs_dhdx,
            nominal_control=self.nominal_control,
            max_control=self.max_control,
            min_control=self.min_control,
            cbf_alpha=self.cbf_alpha,
            slack_penalty=self.slack_penalty,
        )

        solution = multi_cbf_problem.solve(qp_data)

        # Check dimensions with multiple barriers
        self.assertEqual(len(solution.slack), nh)
        self.assertEqual(qp_data.P.shape, (nx + nh, nx + nh))
        self.assertEqual(len(qp_data.l), nx + 2 * nh)


if __name__ == "__main__":
    unittest.main()
