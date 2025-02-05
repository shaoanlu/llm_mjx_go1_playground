from typing import Dict, List, Tuple

from src.estimation.algorithm.base import Filter
from src.estimation.state import State


class StateEstimator:
    """
    Adopted the observer pattern to update the state based on multiple sensor measurements.

    Example usage:
        estimator = StateEstimator()
        estimator.add_filter('imu', ComplementaryFilter(cf_params))
        estimator.add_filter('joint', KalmanFilter(kf_params))
        estimator.add_filter('joint', ParticleFilter(pf_params))
        ...
        state = estimator.update({...})
    """

    def __init__(self, state: State | None = None):
        self._filters: List[Tuple[str, Filter]] = []
        self._state: State | None = state

    def add_filter(self, sensor_type: str, filter: Filter) -> "StateEstimator":
        self._filters.append((sensor_type, filter))
        return self

    def update(self, measurements: Dict[str, Dict]) -> State:
        for sensor_type, filt in self._filters:
            try:
                if sensor_type in measurements:
                    self._state = filt.update(self._state, measurements[sensor_type])
            except Exception as e:
                print(f"Filter update failed for {sensor_type}: {str(e)}")
                continue
        return self._state

    @property
    def state(self) -> State:
        return self._state

    @property
    def filters(self) -> List[Tuple[str, Filter]]:
        return self._filters

    def set_state(self, state: State) -> None:
        self._state = state
