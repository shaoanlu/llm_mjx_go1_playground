from dataclasses import dataclass, field

import numpy as np

from src.control.algorithms.base import Controller, ControllerParams
from src.control.state import Go1State


@dataclass(kw_only=True)
class SequentialControllerParams(ControllerParams):
    yaw_control_threshold: float = field(default=np.pi / 18)
    yaw_control_gain: float = field(default=7.0)
    linear_control_gain: float = field(default=2.0)
    algorithm_type: str = field(default="seq_controller")


class SequentialController(Controller):
    """
    Simple rule-based proportional control
    """

    def __init__(self, config: SequentialControllerParams):
        self.params = config

    def control(self, state: Go1State, ref_state: Go1State, **kwargs) -> np.ndarray:
        x_error, y_error, yaw_error = (ref_state - state).to_array()
        dist = np.linalg.norm([x_error, y_error])

        if abs(yaw_error) > self.params.yaw_control_threshold:
            return np.array([0.0, 0.0, self.params.yaw_control_gain * yaw_error])
        else:
            return np.array([self.params.linear_control_gain * dist, 0.0, 0.0])
