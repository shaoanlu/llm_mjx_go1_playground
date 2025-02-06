from abc import Protocol, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class PlannerParams:
    """Base dataclass for all planner parameters."""

    planner_type: str


class Planner(Protocol):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def plan(self, **kwargs) -> Any: ...
