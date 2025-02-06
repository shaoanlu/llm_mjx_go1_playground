from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class PlannerParams:
    """Base dataclass for all planner parameters."""

    planner_type: str


class Planner(Protocol):
    def __init__(self, config): ...

    @abstractmethod
    def plan(self, **kwargs) -> Any: ...
