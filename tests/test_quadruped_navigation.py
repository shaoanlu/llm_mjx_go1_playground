import logging
import unittest
from pathlib import Path
from typing import List

import jax
import numpy as np

from src.control.algorithms.mlp import MLPPolicyParams
from src.control.controller_factory import ControllerFactory
from src.control.position_controller import (
    PositionControllerParams,
    SequentialControllerParams,
    create_position_controller,
)
from src.environment.env_wrapper import Go1Env
from src.mission_executer import EpisodeResult, MissionConfig, MissionExecuter
from src.planning.base import NavigationPlan


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyPlanner:
    """A planner that only returns a fixed list of waypoints, for debugging purpose"""

    def __init__(self):
        pass

    def plan(self, **kwargs) -> List[np.ndarray]:
        return NavigationPlan(
            waypoints=[np.array([0, 0]), np.array([1, 0]), np.array([1, 3]), np.array([4, 3]), np.array([4, 4])],
            trajectory=[],
        )


class TestQuadrupedNavigation(unittest.TestCase):
    """End-to-end test for LLM-guided quadruped navigation"""

    def setUp(self):
        self.env = Go1Env(env_name="Go1JoystickFlatTerrain")
        self.maze_grid = np.ones((5, 5), dtype=np.int32)
        self.config = MissionConfig(goal=(4, 4), max_sim_steps=1000, retry_delay_sec=0.001, max_attempts=1)

        # Initialize navigation components
        self.planner = DummyPlanner()

        # Initialize controllers
        self.command_generator = self._setup_command_generator()
        self.command_follower = self._setup_command_follower()

    def _setup_command_generator(self):
        """Initialize position controller"""
        factory = ControllerFactory()
        config = PositionControllerParams(
            primary_controller=SequentialControllerParams(),
            fallback_controller=SequentialControllerParams(),
        )
        return create_position_controller(factory, config)

    def _setup_command_follower(self):
        """Initialize MLP policy controller"""
        factory = ControllerFactory()
        mlp_params = MLPPolicyParams.from_dict(
            {"algorithm_type": "mlp", "npy_path": "src/control/nn_params/Go1JoystickFlatTerrain"}
        )
        return factory.build(params=mlp_params)

    def test_navigation_success(self):
        """Test that robot can successfully navigate to goal"""

        mission = MissionExecuter(config=self.config, instruction_prompt="Dummy instruction for integration test.")

        # Run navigation mission
        result = mission.execute_mission(
            planner=self.planner, execute_single_attempt=self.run_episode, rng=jax.random.PRNGKey(0)
        )

        # Verify success
        self.assertEqual(result.status, "Success")
        self.assertGreater(len(result.position_history), 0)

        # Verify final position matches goal
        final_pos = result.position_history[-1]
        np.testing.assert_array_equal(final_pos, np.array(self.config.goal))

    def run_episode(self, waypoints: List[np.ndarray], max_sim_steps: int, rng: jax.Array) -> EpisodeResult:
        """Run single navigation episode"""
        rollout = []
        position_history = []
        waypoint_idx = 0

        # Reset environment
        state = self.env.reset(rng)

        # Run episode
        for _ in range(max_sim_steps):
            # Get next action
            target_pos = waypoints[waypoint_idx] + 0.5
            pos_command = self.command_generator.compute_command(state, target_pos)
            state.info["command"] = pos_command.command

            # Step environment
            action = self.command_follower.control(state.obs["state"])
            state = self.env.step(state, action)

            # Record history
            rollout.append(state)
            curr_pos = pos_command.info.pos[:2]
            if not position_history or not np.array_equal(position_history[-1], curr_pos.astype(np.int32)):
                position_history.append(curr_pos.astype(np.int32))

            # Check termination conditions
            if not self.is_valid_position(curr_pos):
                return EpisodeResult(status="Stop", position_history=position_history, rollout=rollout)

            if pos_command.info.is_arrived:
                waypoint_idx = min(waypoint_idx + 1, len(waypoints))
                if waypoint_idx == len(waypoints):
                    return EpisodeResult(status="Success", position_history=position_history, rollout=rollout)

        return EpisodeResult(status="Timeout", position_history=position_history, rollout=rollout)

    def is_valid_position(self, position: np.ndarray) -> bool:
        """Check if position is valid in maze"""
        pos_int = position.astype(np.int32)
        if not (0 <= pos_int[0] < self.maze_grid.shape[0] and 0 <= pos_int[1] < self.maze_grid.shape[1]):
            return False
        return bool(self.maze_grid[pos_int[0], pos_int[1]])


if __name__ == "__main__":
    unittest.main()
