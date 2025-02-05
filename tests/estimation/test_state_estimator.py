import unittest
from typing import Dict

import numpy as np

from src.estimation.algorithm.base import Filter, FilterParams
from src.estimation.state import State
from src.estimation.state_estimator import StateEstimator


class DummyFilterParams(FilterParams):
    def __init__(self, algorithm_type: str = "dummy"):
        super().__init__(algorithm_type=algorithm_type)


class DummyFilter(Filter):
    """A dummy filter that returns the state unchanged"""

    def __init__(self, params: FilterParams):
        super().__init__(params)

    def update(self, state: State, measurement: Dict) -> State:
        self.state = state
        return state


class AddOneFilter(Filter):
    """A filter that adds 1 to all state values"""

    def __init__(self, params: FilterParams):
        super().__init__(params)

    def update(self, state: State, measurement: Dict) -> State:
        self.state = State(x=state.x + 1, xd=state.xd + 1, info=state.info.copy())
        return self.state


class TestStateEstimator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.initial_state = State(x=np.array([1.0, 2.0, 3.0]), xd=np.array([0.1, 0.2, 0.3]), info={"timestamp": 0})

    def test_initialization(self):
        """Test the initialization of StateEstimator"""
        estimator = StateEstimator()
        self.assertIsNone(estimator.state)
        self.assertEqual(len(estimator.filters), 0)

    def test_add_filter(self):
        """Test adding filters to the estimator"""
        dummy_filter = DummyFilter(DummyFilterParams())
        estimator = StateEstimator(state=self.initial_state)

        estimator.add_filter("imu", dummy_filter)
        self.assertEqual(len(estimator.filters), 1)
        self.assertEqual(estimator.filters[0][0], "imu")
        self.assertIsInstance(estimator.filters[0][1], DummyFilter)

        estimator.add_filter("joint", dummy_filter)
        self.assertEqual(len(estimator.filters), 2)
        self.assertEqual(estimator.filters[1][0], "joint")
        self.assertIsInstance(estimator.filters[1][1], DummyFilter)

    def test_multiple_filters(self):
        """Test adding and using multiple filters"""
        estimator = StateEstimator(state=self.initial_state)
        dummy_filter = DummyFilter(DummyFilterParams())
        add_one_filter = AddOneFilter(DummyFilterParams())
        estimator.add_filter("imu", dummy_filter)
        estimator.add_filter("joint", add_one_filter)

        measurements = {"imu": {"data": [1, 2, 3]}, "joint": {"data": [4, 5, 6]}}

        updated_state = estimator.update(measurements)

        # Since measurements contain both imu and joint data,
        # both filters should be applied in order
        # First the dummy filter (no change), then add_one_filter (+1)
        expected_x = self.initial_state.x + 1
        expected_xd = self.initial_state.xd + 1

        np.testing.assert_array_equal(updated_state.x, expected_x)
        np.testing.assert_array_equal(updated_state.xd, expected_xd)

    def test_update_with_missing_measurements(self):
        """Test updating with measurements missing for some filters"""
        estimator = StateEstimator(state=self.initial_state)
        dummy_filter = DummyFilter(DummyFilterParams())
        add_one_filter = AddOneFilter(DummyFilterParams())
        estimator.add_filter("imu", dummy_filter)
        estimator.add_filter("joint", add_one_filter)

        # Only provide imu measurements
        measurements = {"imu": {"data": [1, 2, 3]}}

        updated_state = estimator.update(measurements)

        # Only dummy_filter should be applied (no change to state)
        np.testing.assert_array_equal(updated_state.x, self.initial_state.x)
        np.testing.assert_array_equal(updated_state.xd, self.initial_state.xd)

    def test_set_state(self):
        """Test setting a new state"""
        estimator = StateEstimator(state=self.initial_state)
        new_state = State(x=np.array([4.0, 5.0, 6.0]), xd=np.array([0.4, 0.5, 0.6]), info={"timestamp": 1})

        estimator.set_state(new_state)

        np.testing.assert_array_equal(estimator.state.x, new_state.x)
        np.testing.assert_array_equal(estimator.state.xd, new_state.xd)
        self.assertEqual(estimator.state.info, new_state.info)

    def test_empty_measurements(self):
        """Test updating with empty measurements"""
        estimator = StateEstimator(state=self.initial_state)
        dummy_filter = DummyFilter(DummyFilterParams())
        estimator.add_filter("imu", dummy_filter)

        updated_state = estimator.update({})

        # State should remain unchanged
        np.testing.assert_array_equal(updated_state.x, self.initial_state.x)
        np.testing.assert_array_equal(updated_state.xd, self.initial_state.xd)


if __name__ == "__main__":
    unittest.main()
