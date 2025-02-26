from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.control.state import Go1State


@dataclass(kw_only=True)
class ControlAffineSystemParams:
    """Base dataclass for all control affine system parameters."""

    x_dim: int
    u_dim: int


class ControlAffineSystem(ABC):
    """
    Interface for a model of a control affine system.

    A control affine system is defined by the following equations:
        x_dot = f(x) + g(x) * u

    where x is the state (of dim n), u is the control input (of dim m), and x_dot is the state derivative.
    """

    def __init__(self, config: ControlAffineSystemParams, **kwargs) -> None:
        self.config = config

    @property
    def x_dim(self) -> int:
        return self.config.x_dim

    @property
    def u_dim(self) -> int:
        return self.config.u_dim

    def forward(self, x: np.ndarray, u: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self.x_dot(x, u)

    def x_dot(self, x: np.ndarray, u: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self.f_x(x) @ x + self.g_x(x) @ u

    def _update_state(self, x: np.ndarray) -> None:
        self.x = x

    @abstractmethod
    def h(self, *args, **kwargs):
        # barrier function
        # R^n -> R^1
        raise NotImplementedError

    @abstractmethod
    def h_dot(self, *args, **kwargs):
        # derivative of the barrier function
        raise NotImplementedError

    @abstractmethod
    def f_x(self, x: Any) -> Any:
        # f(x) of a control affine system: x_dot = f(x) + g(x) * u
        # R^n -> R^n
        raise NotImplementedError

    @abstractmethod
    def g_x(self, x: Any) -> Any:
        # g(x) of a control affine system: x_dot = f(x) + g(x) * u
        # R^n -> R^n*m
        raise NotImplementedError

    @abstractmethod
    def preprocess_go1_state(self, state: Go1State, **kwargs) -> np.ndarray:
        # Extract necessary info from the Go1State and return a numpy array of shape (x_dim,)
        # which should be able to be passed to the argument x of the member functions
        raise NotImplementedError
