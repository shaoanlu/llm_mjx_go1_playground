from dataclasses import dataclass
from typing import Any, Dict

from src.estimation.algorithm.base import Filter, FilterParams, FilterParamsBuilder
from src.estimation.state import State
from src.utils import load_dataclass_from_dict


@dataclass(kw_only=True)  # Make all following fields keyword-only
class ParticleFilterParams(FilterParams):
    num_particles: int
    algorithm_type: str = "particle_filter"


class ParticleFilterParamsBuilder(FilterParamsBuilder):
    def build(self, config: Dict[str, Any]) -> ParticleFilterParams:
        return load_dataclass_from_dict(dataclass=ParticleFilterParams, data_dict=config, convert_list_to_array=False)


class ParticleFilter(Filter):
    def __init__(self, params: FilterParams):
        super().__init__(params)
        self.num_particles = params.num_particles
        self.particles = None

    def update(self, state: State, measurement: Dict) -> State:
        # Particle filter implementation
        return state
