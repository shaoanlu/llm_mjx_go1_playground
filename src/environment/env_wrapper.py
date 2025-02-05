from functools import partial
from typing import Callable

import jax
import numpy as np
from mujoco_playground import registry
from mujoco_playground._src import mjx_env

from src.environment.base import Env


class Go1Env(Env):
    """
    NOTE:
    Go1Handstand environment has observation size of 45
    Go1JoystickFlatTerrain environment has observation size of 48 (45 + 3 commands)
    """

    go1_env_names = ["Go1Handstand", "Go1JoystickFlatTerrain", "Go1Getup", "Go1Footstand"]

    def __init__(self, env_name: str):
        if env_name not in self.go1_env_names:
            raise ValueError(f"Unsupported Go1 environment {env_name}Supported environment are: {self.go1_env_names}")
        self.env_cfg = self.load_config(env_name)
        self.env = registry.load(env_name, config=self.env_cfg)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: mjx_env.State, action: np.ndarray) -> mjx_env.State:
        return self.env.step(state, action)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> mjx_env.State:
        return self.env.reset(rng)

    @staticmethod
    def load_config(env_name: str) -> dict:
        return registry.get_default_config(env_name)

    @property
    def render(self) -> Callable:
        return self.env.render

    @property
    def dt(self) -> float:
        return self.env.dt

    @property
    def get_global_linvel(self) -> jax.Array:
        return self.env.get_global_linvel

    @property
    def get_gyro(self) -> jax.Array:
        return self.env.get_gyro
