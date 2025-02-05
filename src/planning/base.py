from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PlannerParams:
    """Base dataclass for all planner parameters."""

    planner_type: str


class Planner(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def plan(self, env, start, goal):
        raise NotImplementedError
