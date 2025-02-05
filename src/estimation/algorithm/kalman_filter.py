from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from src.estimation.algorithm.base import Filter, FilterParams, FilterParamsBuilder
from src.estimation.state import State
from src.utils import load_dataclass_from_dict


@dataclass(kw_only=True)  # Make all following fields keyword-only
class KalmanFilterParams(FilterParams):
    process_noise: np.ndarray
    measurement_noise: np.ndarray
    algorithm_type: str = "kalman_filter"


class KalmanFilterParamsBuilder(FilterParamsBuilder):
    def build(self, config: Dict[str, Any]) -> KalmanFilterParams:
        return load_dataclass_from_dict(dataclass=KalmanFilterParams, data_dict=config, convert_list_to_array=True)


class KalmanFilter(Filter):
    def __init__(self, params: FilterParams):
        super().__init__(params)
        self.Q = params.process_noise
        self.R = params.measurement_noise

    def update(self, state: State, measurement: Dict) -> State:
        # Kalman filter implementation
        predicted_state = self._predict(state)
        updated_state = self._update(predicted_state, measurement)
        self.state = updated_state
        return self.state

    def _predict(self, state: State) -> State:
        # Predict state
        return state

    def _update(self, state: State, measurement: Dict) -> State:
        # Update state
        return state

    def get_state(self) -> State:
        return self.state
