from dataclasses import dataclass

import numpy as np
from mujoco_playground._src import mjx_env


@dataclass
class Go1State:
    """Represents the state of the robot including position and orientation."""

    position: np.ndarray  # [x, y, z]
    yaw: float

    def __post_init__(self):
        self.position = np.asarray(self.position)  # Ensure position is always a NumPy array

    @classmethod
    def from_mjx_state(cls, state: mjx_env.State) -> "Go1State":
        """Create RobotState from MJX environment state."""
        forward_vec = state.data.site_xmat[1] @ np.array([1.0, 0.0, 0.0])
        yaw = np.arctan2(forward_vec[1], forward_vec[0])
        position = np.array(state.data.site_xpos[0])
        return cls(position=position, yaw=yaw)

    def to_array(self) -> np.ndarray:
        return np.array([self.position[0], self.position[1], self.yaw])

    def __add__(self, other):
        if not isinstance(other, Go1State):
            raise NotImplementedError
        return Go1State(self.position + other.position, self.yaw + other.yaw)

    def __sub__(self, other):
        if not isinstance(other, Go1State):
            return NotImplemented
        return Go1State(self.position - other.position, self.yaw - other.yaw)

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Go1State(self.position * scalar, self.yaw * scalar)

    def __truediv__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Go1State(self.position / scalar, self.yaw / scalar)
