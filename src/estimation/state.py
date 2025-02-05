from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass(kw_only=True)  # Make all following fields keyword-only
class State:
    """
    Reference:
        - https://github.com/google/brax/blob/main/brax/base.py
    """

    x: np.ndarray  # the state, e.g. position
    xd: np.ndarray  # the time-derivative of the state, e.g. velocity
    info: Dict = field(default_factory={})  # Additional information (e.g. timestamp, prediction of the state, etc.)
