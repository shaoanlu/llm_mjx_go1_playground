from dataclasses import dataclass, field
from typing import Callable, Dict
from pathlib import Path

import numpy as np

from src.control.algorithms.base import Controller, ControllerParams
from src.utils import load_dataclass_from_dict


@dataclass(kw_only=True)  # Make all following fields keyword-only
class MLPPolicyParams(ControllerParams):
    """Base dataclass for all MLPPolicy parameters."""

    nn_num_layers: int = field(default=4)
    nn_params: Dict = field(default_factory=dict)
    npy_path: str = field(default="src/control/nn_params/Go1Handstand")
    algorithm_type: str = field(default="mlp")

    def __post_init__(self):
        """Handle initialization processing after dataclass creation."""
        self._load_parameters()

    @classmethod
    def from_dict(cls, data: dict, convert_list_to_array=False):
        data = load_dataclass_from_dict(cls, data, convert_list_to_array=convert_list_to_array)
        data._load_parameters()
        return data

    def _load_parameters(self) -> None:
        """Load all neural network parameters."""
        path = Path(self.npy_path)
        # Load normalization params
        self.nn_params["norm_mean"] = np.load(path / "state_mean.npy")
        self.nn_params["norm_std"] = np.load(path / "state_std.npy")

        # Load layer params
        for i in range(self.nn_num_layers):
            self.nn_params[f"hidden_{i}"] = {
                "kernel": np.load(path / f"hidden_{i}_kernel.npy"),
                "bias": np.load(path / f"hidden_{i}_bias.npy"),
            }


class MLPPolicy(Controller):
    """
    A MLP policy that has no dependency on jax library but simple numpy mat_mul.
    The MLP was trained using locomotion tutorial of mujoco_playground

    NOTE:
    Tried to make it a wrapper for mujoco_playground but not very successful.
    The source code is a little bit difficult to understand w/o knowledge of the brax/mujoco structure.

    https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py#L331
    https://github.com/google/brax/blob/main/brax/training/agents/ppo/networks.py#L34
    """

    def __init__(self, params: MLPPolicyParams):
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
                    x = x / (1 + np.exp(-1 * x))  # brax PPO training defaults to swish as the activation func

            # output tanh activation
            x, _ = np.split(x, 2, axis=-1)  # split into loc and scale of a normal distribution
            x = np.tanh(x)
            return x

        self._inference: Callable = forward
