from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Protocol


@dataclass(kw_only=True, frozen=True)
class NavigationPlan:
    waypoints: List[Any]
    trajectory: List[Any]


@dataclass
class PlannerParams:
    """Base dataclass for all planner parameters."""

    planner_type: str


class Planner(Protocol):
    def __init__(self, config): ...

    @abstractmethod
    def plan(self, **kwargs) -> Any: ...


class FixedWaypointsPlanner:
    """
    A simple planner that plans a fixed set of waypoints.
    """

    def __init__(self, waypoints: List[Any]):
        self.waypoints = waypoints

    def plan(self, **kwargs) -> NavigationPlan:
        return NavigationPlan(waypoints=self.waypoints, trajectory=[])
