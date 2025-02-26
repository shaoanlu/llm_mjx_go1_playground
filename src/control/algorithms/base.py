from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from src.control.state import Go1Command
from src.utils import load_dataclass_from_dict


@dataclass(kw_only=True)  # Make all following fields keyword-only
class ControllerParams(ABC):
    """Base dataclass for all controller parameters."""

    algorithm_type: str

    @classmethod
    def from_dict(cls, data: dict, convert_list_to_array=False):
        return load_dataclass_from_dict(cls, data, convert_list_to_array=convert_list_to_array)


class Controller(ABC):
    def __init__(self, config: ControllerParams, **kwargs):
        pass

    @abstractmethod
    def control(self, state: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def reset(self, **kwargs):
        return self.init_control_params


@dataclass(kw_only=True, frozen=True)
class HighlevelControllerInfo:
    info_type: str = "default"


@dataclass(kw_only=True)
class HighLevelCommand:
    value: np.ndarray | jax.Array  # X, Y, yaw
    info: HighlevelControllerInfo

    def __post_init__(self):
        # Enforce value as jax array
        if isinstance(self.value, np.ndarray):
            self.value = jax.numpy.array(self.value)

    def as_go1_command(self) -> Go1Command:
        # Go1Command expect value attr as np.ndarray
        return Go1Command(value=np.array(self.value))


class HighLevelController(ABC):
    """
    Interface for a high-level controller that generates a command for a low-level command follower.
    The instance of this class usually contains one or more low-level controllers.
    """

    def __init__(self, config: ControllerParams, **kwargs):
        pass

    def build_controller(self, **kwargs) -> None:
        raise NotImplementedError

    def compute_command(self, state: Any, **kwargs) -> HighLevelCommand:
        raise NotImplementedError
