from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from src.control.models.base import ControlAffineSystem, ControlAffineSystemParams


@dataclass(kw_only=True)
class Simple2DRobotConfig(ControlAffineSystemParams):
    a: float = field(default=41.0)  # ellipse param approximating robot collision region in XY plane
    b: float = field(default=24.0)  # ellipse param approximating robot collision region in XY plane
    x_dim: int = field(default=2)  # dimension of the state space
    u_dim: int = field(default=2)  # dimension of the control space


class Simple2DRobot(ControlAffineSystem):
    """
    A simple 2D robot model with velocity control.
    This class is intentionally to be stateless: the state x and control u are not stored in the class.

    The system dynamics are defined as:
        x__dot = u
    where x is the state (XY position), u is the control input (of dim 2), and x_dot is the velocity of the robot.

    In addition, the shape of the robot is approximated as a ellipse with a given width and height.
        (x - xc)^2 / a^ + (y - yc)^2 / b^2 = 1
    where (xc, yc) is the center of the ellipse, a is the semi-major axis, and b is the semi-minor
    """

    def __init__(self, config: Simple2DRobotConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def f_x(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((self.config.x_dim, self.config.x_dim))

    def g_x(self, x: np.ndarray) -> np.ndarray:
        return np.eye(self.config.x_dim)

    def h(self, x: np.ndarray, obs_x: np.ndarray) -> np.array:
        """
        Distance between the robot (as an ellipse) and the obstacle
        barrier func is ||x - xr||^2 - dist_collision**2
        where dist_collision is the distance between the center of the ellipse and intersection point on the ellipse
        """
        intersection_points = _calculate_ellipse_closest_point(center=x, a=self.config.a, b=self.config.b, x=obs_x)
        dist_collision = np.linalg.norm(x - intersection_points, axis=1)
        return (np.linalg.norm(x - obs_x, axis=1) ** 2 - dist_collision**2).squeeze()

    def h_dot(self, x: np.ndarray, obs_x: np.ndarray) -> np.array:
        return 2 * (x - obs_x)


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
