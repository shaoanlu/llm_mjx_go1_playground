from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from src.control.models.base import ControlAffineSystem, ControlAffineSystemParams


@dataclass(kw_only=True)
class UnicycleRobotParams(ControlAffineSystemParams):
    a: float = field(default=0.45)  # ellipse param (in meter) approximating robot collision region in XY plane
    b: float = field(default=0.3)  # ellipse param (in meter) approximating robot collision region in XY plane
    x_dim: int = field(default=3)  # dimension of the state space (x, y, theta)
    u_dim: int = field(default=2)  # dimension of the control space (v, omega)


class UnicycleRobot(ControlAffineSystem):
    """
    A Unicycle robot model with velocity control.
    This class is intentionally to be stateless: the state x and control u are not stored in the class.

    The system dynamics are defined as:
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        theta_dot = omega
    where x=[x, y, theta] is the state (XY position and yaw angle),
    u=[v, omega] is the control input (of dim 2), and x_dot is the time derivative of the state.

    In addition, the shape of the robot is approximated as a ellipse with a given width and height.
        (x - xc)^2 / a^ + (y - yc)^2 / b^2 = 1
    where (xc, yc) is the center of the ellipse, a is the semi-major axis, and b is the semi-minor
    """

    def __init__(self, config: UnicycleRobotParams = UnicycleRobotParams(), **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def f_x(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((self.config.x_dim, self.config.x_dim))

    def g_x(self, x: np.ndarray) -> np.ndarray:
        _, _, yaw_angle = x
        return np.array([[np.cos(yaw_angle), 0], [np.sin(yaw_angle), 0], [0, 1]])

    def h(self, x: np.ndarray, obs_x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Reference: https://arxiv.org/abs/2307.08227

        Returns:
            The value of the barrier function. shape=(N,)
        """
        l = np.linalg.norm(x)
        theta = np.arctan2(x[1], x[0])
        new_x = np.array([l * np.cos(theta), l * np.sin(theta)]) - obs_x

        intersection_points = _calculate_ellipse_closest_point(center=x, a=self.config.a, b=self.config.b, x=obs_x)
        dist_collision = np.linalg.norm(x - intersection_points, axis=1)
        return np.linalg.norm(new_x, axis=1) ** 2 - dist_collision**2  # shape=(N,)

    def h_dot(self, x: np.ndarray, obs_x: np.ndarray, **kwargs) -> np.array:
        l = np.linalg.norm(x)
        theta = np.arctan2(x[1], x[0])
        new_x = np.array([l * np.cos(theta), l * np.sin(theta)]) - obs_x

        trans = np.array([[np.cos(theta), np.sin(theta)], [-l * np.sin(theta), l * np.cos(theta)]]).T
        return 2 * new_x @ trans


def _calculate_ellipse_closest_point(center: Tuple | np.ndarray, a: float, b: float, x: np.ndarray) -> np.ndarray:
    """
    Compute the closest XY point on an ellipse to a set of points.

    Args:
        center: The center of the ellipse. shape=(2,)
        a: The length of the semi-major axis.
        b: The length of the semi-minor axis.
        x: The points to find the closest point on the ellipse to. shape=(N, 2)

    Returns:
        The closest points on the ellipse to the input points. shape=(N, 2)
    """
    assert len(center) == 2, "Center must be a tuple or array of shape (2,) representing XY position of the ellipse"
    assert a > 0 and b > 0, f"Ellipse must have positive semi-major and semi-minor axes, {a=}, {b=}"
    c1, c2 = center  # Ellipse center
    x1, x2 = x[:, 0] - c1, x[:, 1] - c2  # Transform to ellipse-centered coordinates

    # Compute angle of the points w.r.t the ellipse center by normalizing it to a unit circle
    theta = np.arctan2(x2 / b, x1 / a)

    # Compute closest points on the ellipse using parametric form
    y1 = a * np.cos(theta) + c1
    y2 = b * np.sin(theta) + c2

    return np.column_stack((y1, y2))
