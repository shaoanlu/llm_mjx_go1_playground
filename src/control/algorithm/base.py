from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(kw_only=True)  # Make all following fields keyword-only
class ControllerParams(ABC):
    """Base dataclass for all controller parameters."""

    algorithm_type: str


class ControllerParamsBuilder(ABC):
    """Abstract base class for parameter builders."""

    @abstractmethod
    def build(self, config: Dict[str, Any]) -> ControllerParams:
        pass


class Controller(ABC):
    def __init__(self, config: ControllerParams, **kwargs):
        pass

    @abstractmethod
    def control(self, state: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def reset(self, **kwargs):
        return self.init_control_params
