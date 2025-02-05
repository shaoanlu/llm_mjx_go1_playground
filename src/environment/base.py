from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class Env(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._init()

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        raise NotImplementedError
