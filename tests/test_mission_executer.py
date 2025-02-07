import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import Mock, patch

import numpy as np
from mujoco_playground._src import mjx_env

from src.mission_executer import EpisodeResult, MissionConfig, MissionExecuter, MissionResult
from src.planning.llm_nagivator import GeminiThinkingNavigator


class TestMissionConfig(unittest.TestCase):
    """Test MissionConfig dataclass"""

    def test_valid_initialization(self):
        """Test initialization with valid parameters"""
        config = MissionConfig(goal=(4, 4), max_sim_steps=1000, retry_delay_sec=5, max_attempts=20)
        self.assertEqual(config.goal, (4, 4), msg=f"{config.goal=}")
        self.assertEqual(config.max_sim_steps, 1000, msg=f"{config.max_sim_steps=}")
        self.assertEqual(config.retry_delay_sec, 5, msg=f"{config.retry_delay_sec=}")
        self.assertEqual(config.max_attempts, 20, msg=f"{config.max_attempts=}")

    def test_invalid_parameters(self):
        """Test initialization with invalid parameters"""
        with self.assertRaises(ValueError):
            MissionConfig(max_sim_steps=-1)

        with self.assertRaises(ValueError):
            MissionConfig(max_attempts=0)

    def test_custom_parameters(self):
        """Test initialization with custom parameters"""
        config = MissionConfig(goal=(10, 10), max_sim_steps=500, retry_delay_sec=2, max_attempts=5)
        self.assertEqual(config.goal, (10, 10), msg=f"{config.goal=}")
        self.assertEqual(config.max_sim_steps, 500, msg=f"{config.max_sim_steps=}")
        self.assertEqual(config.retry_delay_sec, 2, msg=f"{config.retry_delay_sec=}")
        self.assertEqual(config.max_attempts, 5, msg=f"{config.max_attempts=}")


class TestMissionResult(unittest.TestCase):
    """Test MissionResult dataclass"""

    def setUp(self):
        self.position_history = [np.array([0, 0]), np.array([1, 1])]
        self.waypoints = [np.array([0, 0]), np.array([2, 2])]
        self.rollouts = [Mock(spec=mjx_env.State) for _ in range(2)]

    def test_successful_result(self):
        """Test creation of successful mission result"""
        result = MissionResult(
            status="Success",
            position_history=self.position_history,
            message="Mission completed",
            waypoints=self.waypoints,
            rollouts=self.rollouts,
        )
        self.assertEqual(result.status, "Success", msg=f"{result.status=}")
        self.assertEqual(result.message, "Mission completed", msg=f"{result.message=}")
        self.assertEqual(len(result.position_history), 2, msg=f"{result.position_history=}")
        self.assertEqual(len(result.waypoints), 2, msg=f"{result.waypoints=}")
        self.assertEqual(len(result.rollouts), 2, msg=f"{result.rollouts=}")

    def test_failed_result(self):
        """Test creation of failed mission result"""
        result = MissionResult(
            status="Failed: Max attempts reached",
            position_history=self.position_history,
            message="Max attempts exceeded",
            waypoints=self.waypoints,
            rollouts=self.rollouts,
        )
        self.assertEqual(result.status, "Failed: Max attempts reached", f"{result.status=}")
        self.assertEqual(result.message, "Max attempts exceeded", f"{result.message=}")

    def test_immutability(self):
        """Test that MissionResult is immutable"""
        result = MissionResult(
            status="Success",
            position_history=self.position_history,
            waypoints=self.waypoints,
            rollouts=self.rollouts,
        )
        with self.assertRaises(FrozenInstanceError):
            result.status = "Failed"


class TestEpisodeResult(unittest.TestCase):
    """Test EpisodeResult dataclass"""

    def setUp(self):
        self.position_history = [np.array([0, 0]), np.array([1, 1])]
        self.rollout = [Mock(spec=mjx_env.State) for _ in range(2)]

    def test_valid_status_values(self):
        """Test creation with valid status values"""
        valid_statuses = ["Stop", "Timeout", "Success"]
        for status in valid_statuses:
            result = EpisodeResult(
                status=status, position_history=self.position_history, rollout=self.rollout
            )
            self.assertEqual(result.status, status, msg=f"{result.status=}")

    def test_immutability(self):
        """Test that EpisodeResult is immutable"""
        result = EpisodeResult(
            status="Success", position_history=self.position_history, rollout=self.rollout
        )
        with self.assertRaises(FrozenInstanceError):
            result.status = "Stop"


class TestMissionExecuter(unittest.TestCase):
    """Test MissionExecuter class"""

    def setUp(self):
        self.config = MissionConfig(
            goal=(4, 4),
            max_sim_steps=1000,
            retry_delay_sec=0.001,
            max_attempts=3,
        )
        self.instruction_prompt = "Navigate to goal"
        self.executer = MissionExecuter(self.config, self.instruction_prompt)
        self.mock_planner = Mock(spec=GeminiThinkingNavigator)
        self.mock_execute_single_attempt = Mock()

    def test_initialization(self):
        """Test MissionExecuter initialization"""
        self.assertEqual(
            self.executer.instruction_prompt,
            self.instruction_prompt,
            msg=f"{self.executer.instruction_prompt=}",
        )
        self.assertIsNone(self.executer.current_position, msg=f"{self.executer.current_position=}")
        self.assertEqual(self.executer.waypoints, [], msg=f"{self.executer.waypoints=}")
        self.assertEqual(
            self.executer.position_history, [], msg=f"{self.executer.position_history=}"
        )

    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_successful_mission_execution(self, mock_sleep):
        """Test successful mission execution"""
        # Setup mock returns
        waypoints = [np.array([0, 0]), np.array([2, 2]), np.array([4, 4])]
        self.mock_planner.plan.return_value = waypoints

        position_history = [np.array([0, 0]), np.array([4, 4])]
        self.mock_execute_single_attempt.return_value = EpisodeResult(
            status="Success", position_history=position_history, rollout=[Mock(spec=mjx_env.State)]
        )

        result = self.executer.execute_mission(
            self.mock_planner, self.mock_execute_single_attempt, print_result=False
        )

        self.assertEqual(result.status, "Success", msg=f"{result.status=}")
        self.assertEqual(len(result.position_history), 2, msg=f"{result.position_history=}")
        self.mock_planner.plan.assert_called_once()
        self.mock_execute_single_attempt.assert_called_once()

    @patch("time.sleep")
    def test_failed_mission_execution(self, mock_sleep):
        """Test failed mission execution (max attempts reached)"""
        # Setup mock returns
        waypoints = [np.array([0, 0]), np.array([2, 2])]
        self.mock_planner.plan.return_value = waypoints

        position_history = [np.array([0, 0]), np.array([1, 1])]
        self.mock_execute_single_attempt.return_value = EpisodeResult(
            status="Stop", position_history=position_history, rollout=[Mock(spec=mjx_env.State)]
        )

        result = self.executer.execute_mission(
            self.mock_planner, self.mock_execute_single_attempt, print_result=False
        )

        self.assertEqual(result.status, "Failed: Max attempts reached", msg=f"{result.status=}")
        self.assertEqual(
            self.mock_planner.plan.call_count,
            self.config.max_attempts,
            msg=f"{self.mock_planner.plan.call_count=}",
        )
        self.assertEqual(
            self.mock_execute_single_attempt.call_count,
            self.config.max_attempts,
            msg=f"{self.mock_execute_single_attempt.call_count=}",
        )

    def test_waypoint_validation(self):
        """Test waypoint validation logic"""
        waypoints = [np.array([0, 0]), np.array([2, 2])]  # Does not contain goal position (4, 4)
        validated = self.executer._validate_waypoints(list(waypoints))  # Input a copy of waypoints

        self.assertEqual(len(validated), len(waypoints) + 1, msg=f"{validated=}")
        np.testing.assert_array_equal(
            validated[-1], np.array(self.config.goal), err_msg=f"{validated[-1]=}"
        )

    def test_failure_message_formatting(self):
        """Test failure message formatting"""
        position_history = [np.array([0, 0]), np.array([1, 1])]
        result = EpisodeResult(
            status="Stop", position_history=position_history, rollout=[Mock(spec=mjx_env.State)]
        )

        message = self.executer._format_failure_message(result)
        self.assertIn("Failed: Stop", message, msg=f"{message=}")
        self.assertIn("(1, 1)", message, msg=f"{message=}")

    def test_invalid_failure_type(self):
        """Test handling of invalid failure type"""
        position_history = [np.array([0, 0])]
        result = EpisodeResult(
            status="InvalidStatus",  # type: ignore
            position_history=position_history,
            rollout=[Mock(spec=mjx_env.State)],
        )

        with self.assertRaises(ValueError):
            self.executer._format_failure_message(result)


if __name__ == "__main__":
    unittest.main()
