from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np

from src.control.algorithm.base import Controller, ControllerParams, ControllerParamsBuilder


@dataclass(kw_only=True)  # Make all following fields keyword-only
class PPOParams(ControllerParams):
    """Base dataclass for all PPO parameters."""

    nn_num_layers: int
    nn_params: Dict
    algorithm_type: str = field(default="ppo")


class PPOParamsBuilder(ControllerParamsBuilder):
    """
    Mostly hard-coded build function, for simplicity.
    """

    def build(self, config: Dict = None) -> PPOParams:
        if config is not None:
            npy_path = config["npy_path"]
        else:
            npy_path = "examples/mujoco_Go1/nn_params/Go1Handstand"
        num_layers = 4
        nn_params = {}
        nn_params["norm_mean"] = np.load(f"{npy_path}/state_mean.npy")
        nn_params["norm_std"] = np.load(f"{npy_path}/state_std.npy")
        for i in range(num_layers):  # [0, 1, 2, 3]
            nn_params[f"hidden_{i}"] = {}
            nn_params[f"hidden_{i}"]["kernel"] = np.load(f"{npy_path}/hidden_{i}_kernel.npy")
            nn_params[f"hidden_{i}"]["bias"] = np.load(f"{npy_path}/hidden_{i}_bias.npy")
        return PPOParams(nn_num_layers=num_layers, nn_params=nn_params, algorithm_type="ppo")


class PPO(Controller):
    """
    A PPO policy that has no dependency on jax library but simple numpy mat_mul.
    The MLP was trained using locomotion tutorial of mujoco_playground

    NOTE:
    Tried to make it a wrapper for mujoco_playground but not very successful.
    The source code is a little bit difficult to understand w/o knowledge of the brax/mujoco structure.

    https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py#L331
    https://github.com/google/brax/blob/main/brax/training/agents/ppo/networks.py#L34
    """

    def __init__(self, params: PPOParams):
        self.params = params
        self.build_network()

    def control(self, state: np.ndarray, **kwargs) -> np.ndarray:
        nn_inp = np.array(state.copy())
        return self._inference(nn_inp)

    def build_network(self) -> None:
        params = self.params

        def forward(x: np.ndarray):
            """
            Normalization and forward pass of a MLP
            """
            # normalize input
            x = (x - params.nn_params["norm_mean"]) / params.nn_params["norm_std"]

            # forward pass
            for i in range(params.nn_num_layers):
                x = params.nn_params[f"hidden_{i}"]["kernel"].T @ x + params.nn_params[f"hidden_{i}"]["bias"]
                if i < (params.nn_num_layers - 1):
                    x = x / (1 + np.exp(-1 * x))  # brax PPO defaults to swish as the activation func

            # output tanh activation
            x, _ = np.split(x, 2, axis=-1)  # split into loc and scale of a normal distribution
            x = np.tanh(x)
            return x

        self._inference: Callable = forward
