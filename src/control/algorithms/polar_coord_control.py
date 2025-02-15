from dataclasses import dataclass, field

import numpy as np

from src.control.algorithms.base import Controller, ControllerParams
from src.control.state import Go1State


@dataclass(kw_only=True)
class PolarCoordinateControllerParams(ControllerParams):
    # make sure yaw_control_gain1 - linear_control_gain > yaw_control_gain2 for local stability
    linear_control_gain: float = field(default=2.0)
    yaw_control_gain1: float = field(default=7.0)
    yaw_control_gain2: float = field(default=1.0)
    rho_threshold: float = field(default=0.1)
    algorithm_type: str = field(default="polar_coord_controller")


class PolarCoordinateController(Controller):
    """
    http://www.cs.cmu.edu/~rasc/Download/AMRobots3.pdf
    https://doi.org/10.1177/1729881418806435
    """

    def __init__(self, config: PolarCoordinateControllerParams):
        self.params = config

    def control(self, state: Go1State, ref_state: Go1State, **kwargs) -> np.ndarray:
        # Error calculation
        tar_yaw, yaw = np.unwrap([ref_state.yaw, state.yaw])
        error_x = ref_state.position[0] - state.position[0]
        error_y = ref_state.position[1] - state.position[1]
        error_yaw = tar_yaw - yaw

        # Polar coordinate transformation
        rho = np.sqrt(error_x**2 + error_y**2)
        alpha = np.arctan2(error_y, error_x) - yaw
        beta = alpha + yaw

        # Control inputs
        k1 = self.params.linear_control_gain
        k2 = self.params.yaw_control_gain1
        k3 = self.params.yaw_control_gain2
        v = k1 * rho * np.cos(alpha)
        if rho >= self.params.rho_threshold:  # prevent sigularity
            omega = k2 * alpha + k1 * (np.sin(alpha) * np.cos(alpha) / alpha) * (alpha + k3 * beta)
        else:
            omega = k2 * error_yaw
        return np.array([v, 0.0, omega])
