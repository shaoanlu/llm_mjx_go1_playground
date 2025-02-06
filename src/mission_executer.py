from dataclasses import dataclass
from typing import Callable, ClassVar, Tuple, List, Literal
import numpy as np
from mujoco_playground._src import mjx_env
import time

from src.environment.traversability_map import TraversabilityMap
from src.environment.env_wrapper import Go1Env
from src.planning.llm_nagivator import GeminiThinkingNavigator


@dataclass(kw_only=True, frozen=True)
class MissionConfig:
    goal: Tuple[int, int] = (4, 4)  # Target destination coordinates (x, y)
    max_sim_steps: int = 30  # Maximum number of steps allowed per attempt
    retry_delay_sec: int = 5  # elay between retry attempts to respect API limits
    max_attempts: int = 20  # Maximum number of mission retry attempts

    def __post_init__(self):
        if self.max_steps <= 0 or self.max_attempts <= 0:
            raise ValueError(
                f"Steps and attempts must be positive, {self.max_steps=}, {self.max_attempts=}"
            )
        if self.position_bounds[0] >= self.position_bounds[1]:
            raise ValueError(f"Invalid position bounds, {self.position_bounds=}")


@dataclass
class MissionResult(frozen=True):
    """Represents the outcome of a mission execution."""

    status: Literal["Success", "Failed: Max attempts reached"]
    position_history: List[np.ndarray]
    message: str = ""
    waypoints: List[np.ndarray]
    rollouts: List[mjx_env.State]


@dataclass(frozen=True)
class EpisodeResult:
    """Result of a full episode."""

    status: Literal["Stop", "Timeout", "Success"]
    position_history: List[np.ndarray]
    rollout: List


class MissionExecuter:
    """Controls the execution of navigation missions using LLM-guided waypoints.

    Coordinates between grid management, navigation, and LLM components to execute
    complete navigation missions with retry logic and position tracking.
    """

    FAILURE_TYPES: ClassVar[tuple[str, ...]] = ("Stop", "Timeout", "Max attempts reached")

    def __init__(self, config: MissionConfig, instruciton_prompt: str):
        self.config = config
        self.instruction_prompt: str = instruciton_prompt
        self.current_position: np.ndarray | None = None
        self.waypoints: List[Tuple] = []
        self.position_history = []

    def execute_mission(
        self,
        planner: GeminiThinkingNavigator,
        execute_single_attempt: Callable,
    ) -> Tuple[str, List[np.ndarray]]:
        """
        Execute complete navigation mission.

        Returns:
            str: Mission status
            List[np.ndarray]: List of traversed XY positions
        """
        rollout_of_all_attempts = []

        # Run the mission for a number of attempts, each time with a different init position and prompt
        for attempt in range(self.config.max_attempts):
            # Prompt the LLM to get waypoints suggestion
            if attempt == 0:
                prompt: str = (
                    self.instruction_prompt + f"\nStart. you are at somwwhere near (0, 0)."
                )
            waypoints = planner.plan(prompt=prompt)
            waypoints = self._validate_waypoints(waypoints)

            # run the mission (simulation)
            result: EpisodeResult = execute_single_attempt(
                waypoints=waypoints,
                max_sim_steps=self.config.max_sim_steps,
            )

            # print debug information
            print(f"[Trial {attempt + 1}]\n{prompt=}\n{waypoints=}\n{result=}\n")

            if result.status == "Success":
                return MissionResult(
                    status=result.status,
                    position_history=result.position_history,
                    message="",
                    waypoints=waypoints,
                )

            prompt = self._format_failure_message(result)
            rollout_of_all_attempts.extend(result.rollout)

            # add a delay before retrying to avoid API rate limiting
            time.sleep(self.config.retry_delay_sec)

        return self._create_failure_status(
            MissionResult(
                status="Max attempts reached",
                position_history=result.position_history,
                message="",
                waypoints=waypoints,
            )
        )

    def _validate_waypoints(self, waypoints: List[np.ndarray]) -> List[np.ndarray]:
        goal_position = np.array(self.config.goal)
        if np.array_equal(waypoints[-1], goal_position):
            waypoints.append(goal_position)
        return waypoints

    def _create_failure_status(self, result: EpisodeResult) -> MissionResult:
        """Create formatted failure status with position information."""

        message = self._format_failure_message(result)

        return MissionResult(
            status=f"Failed: {result.status}",
            position_history=result.position_history,
            message=message,
        )

    def _format_failure_message(self, result: EpisodeResult) -> str:
        """Format failure message with position history."""
        if result.status not in self.FAILURE_TYPES:
            raise ValueError(f"Invalid failure type: {result.status}")
        x, y = result.position_history[-1].astype(int)
        return f"Failed: {result.status} at ({x}, {y}), traversed cells: {[tuple(np.round(x_, 1)) for x_ in result.position_history]}"
