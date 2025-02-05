from dataclasses import dataclass
from typing import Any, Dict

from src.estimation.algorithm.base import Filter, FilterParams, FilterParamsBuilder
from src.estimation.state import State
from src.utils import load_dataclass_from_dict


@dataclass(kw_only=True)  # Make all following fields keyword-only
class ComplementaryFilterParams(FilterParams):
    alpha: float
    algorithm_type: str = "complementary_filter"


class ComplementaryFilterParamsBuilder(FilterParamsBuilder):
    def build(self, config: Dict[str, Any]) -> ComplementaryFilterParams:
        return load_dataclass_from_dict(
            dataclass=ComplementaryFilterParams, data_dict=config, convert_list_to_array=False
        )


class ComplementaryFilter(Filter):
    def __init__(self, params: FilterParams):
        super().__init__(params)
        self.alpha = params.alpha

    def update(self, state: State, measurement: Dict) -> State:
        # Complementary filter implementation
        return state
