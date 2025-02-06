from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

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
