import time
from dataclasses import dataclass
from typing import Callable, ClassVar, List, Literal, Tuple

import jax
import numpy as np
from mujoco_playground._src import mjx_env

from src.planning.base import NavigationPlan
from src.planning.llm_nagivator import GeminiThinkingNavigator
from src.utils import load_dataclass_from_dict


@dataclass(kw_only=True)
class MissionConfig:
    goal: Tuple[int, int] = (4, 4)  # Target destination coordinates (x, y)
    max_sim_steps: int = 1000  # Maximum number of steps allowed per attempt
    retry_delay_sec: int = 5  # elay between retry attempts to respect API limits
    max_attempts: int = 20  # Maximum number of mission retry attempts

    def __post_init__(self):
        if self.max_sim_steps <= 0 or self.max_attempts <= 0:
            raise ValueError(f"Steps and attempts must be positive, {self.max_sim_steps=}, {self.max_attempts=}")

    @classmethod
    def from_dict(cls, data: dict, convert_list_to_array=False):
        return load_dataclass_from_dict(cls, data, convert_list_to_array=convert_list_to_array)


@dataclass(kw_only=True, frozen=True)
class MissionResult:
    """Represents the outcome of a mission execution."""

    status: Literal["Success", "Failed: Max attempts reached"]
    position_history: List[np.ndarray]
    message: str = ""
    waypoints: List[np.ndarray]
    rollouts: List[mjx_env.State]


@dataclass(kw_only=True, frozen=True)
class EpisodeResult:
    """Result of a full episode."""

    status: Literal["Stop", "Timeout", "Success"]
    position_history: List[np.ndarray]
    rollout: List[mjx_env.State]


class MissionExecuter:
    """Controls the execution of navigation missions using LLM-guided waypoints.
    Coordinates between grid management, navigation, and LLM components to execute
    complete navigation missions with retry logic and position tracking.

    This class is responsible for interactions with the LLM navigation planner,
    while keeping itself agnostic to the specifics of the navigation mission.
    """

    SUCCESS_TYPES: ClassVar[tuple[str, ...]] = ("Success",)
    FAILURE_TYPES: ClassVar[tuple[str, ...]] = ("Stop", "Timeout", "Max attempts reached")

    def __init__(self, config: MissionConfig, instruction_prompt: str):
        self.config = config
        self.instruction_prompt: str = instruction_prompt
        self.current_position: np.ndarray | None = None
        self.waypoints: List[Tuple] = []
        self.position_history = []

    def execute_mission(
        self,
        planner: GeminiThinkingNavigator,
        execute_single_attempt: Callable,
        rng: jax.Array,
        print_result: bool = True,
    ) -> Tuple[str, List[np.ndarray]]:
        """
        Execute complete navigation mission.

        Args:
            execute_single_attempt: A function that accepts arguments waypoints, max_sim_steps,
                and a random key as input and return an object of EpisodeResult.

        Returns:
            str: Mission status
            List[np.ndarray]: List of traversed XY positions
        """
        rollout_of_all_attempts = []

        # Run the mission for a number of attempts, each time with a different init position and prompt
        for attempt in range(self.config.max_attempts):
            # Prompt the LLM to get waypoints suggestion
            if attempt == 0:
                prompt: str = self.instruction_prompt + "\nStart. you are at somwwhere near (0, 0)."
            nav_plan: NavigationPlan = planner.plan(prompt=prompt)
            waypoints = list(nav_plan.waypoints)
            waypoints = self._validate_waypoints(waypoints)

            # run the mission (simulation)
            _, rng = jax.random.split(rng)
            result: EpisodeResult = execute_single_attempt(
                waypoints=waypoints,
                max_sim_steps=self.config.max_sim_steps,
                rng=rng,
            )
            rollout_of_all_attempts.extend(result.rollout)

            # print debug information
            if print_result:
                print(
                    f"[Trial {attempt + 1}]\n{prompt=}\n{waypoints=}\n{result.status=}{tuple(result.position_history[-1])}",
                    f"\t{result.position_history=}\n",
                )

            if result.status in self.SUCCESS_TYPES:
                return MissionResult(
                    status=result.status,
                    position_history=result.position_history,
                    message=prompt,
                    waypoints=waypoints,
                    rollouts=rollout_of_all_attempts,
                )

            prompt = self._format_failure_message(result)

            # add a delay before retrying to avoid API rate limiting
            time.sleep(self.config.retry_delay_sec)

        return MissionResult(
            status="Failed: Max attempts reached",
            position_history=result.position_history,
            message=prompt,
            waypoints=waypoints,
            rollouts=rollout_of_all_attempts,
        )

    def _validate_waypoints(self, waypoints: List[np.ndarray]) -> List[np.ndarray]:
        """Validate waypoints by ensuring the last waypoint is the goal position."""
        goal_position = np.array(self.config.goal)
        if not np.array_equal(waypoints[-1], goal_position):
            waypoints.append(goal_position)
        return waypoints

    def _format_failure_message(self, result: EpisodeResult) -> str:
        """Format failure message with position history."""
        if result.status not in self.FAILURE_TYPES:
            raise ValueError(f"Invalid failure type: {result.status}")
        x, y = result.position_history[-1].astype(int)
        return f"Failed: {result.status} at ({x}, {y}), traversed cells: {[tuple(np.round(x_, 1)) for x_ in result.position_history]}"
