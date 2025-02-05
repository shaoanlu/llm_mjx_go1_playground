import unittest
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from src.control.algorithm.base import Controller, ControllerParams, ControllerParamsBuilder
from src.control.controller_factory import ConfigFactory, ControllerFactory


# Create dummy classes for testing


@dataclass(kw_only=True)  # Make all following fields keyword-only
class DummyParams(ControllerParams):
    value: int
    algorithm_type: str = field(default="dummy")


class DummyControllerParamsBuilder(ControllerParamsBuilder):
    @staticmethod
    def build(config: Dict[str, Any]) -> DummyParams:
        return DummyParams(value=config.get("value", 0))


class DummyController(Controller):
    def __init__(self, params):
        self.params = params

    def control(self, state: np.ndarray, **kwargs) -> np.ndarray:
        pass


class TestFactories(unittest.TestCase):
    def setUp(self):
        self.config_factory = ConfigFactory()
        self.controller_factory = ControllerFactory()

        # Add dummy classes to the factories' maps
        self.config_factory.register_config("dummy", DummyControllerParamsBuilder)
        self.controller_factory.register_controller(DummyParams, DummyController)
        self.controller_factory.config_factory = self.config_factory

    def test_config_factory_with_dummy(self):
        # Test building dummy params
        dummy_config = {"algorithm_type": "dummy", "value": 42}
        params = self.config_factory.build(dummy_config)
        self.assertIsInstance(params, DummyParams)
        self.assertEqual(params.value, 42)
        self.assertEqual(params.algorithm_type, "dummy")

    def test_config_factory_invalid_type(self):
        # Test invalid algorithm type
        invalid_config = {"algorithm_type": "invalid"}
        with self.assertRaises(ValueError):
            self.config_factory.build(invalid_config)

    def test_controller_factory_with_dummy(self):
        # Test building dummy controller from params
        params = DummyParams(value=42)
        controller = self.controller_factory.build(params)
        self.assertIsInstance(controller, DummyController)

        # Test building dummy controller from dict
        params_dict = {"value": 42, "algorithm_type": "dummy"}
        controller = self.controller_factory.build_from_dict(params_dict)
        self.assertIsInstance(controller, DummyController)

    def test_controller_factory_invalid_params(self):
        # Test invalid parameter type

        @dataclass
        class InvalidParams(ControllerParams):
            algorithm_type: str = "invalid_params"

        invalid_params = InvalidParams()
        with self.assertRaises(ValueError):
            self.controller_factory.build(invalid_params)


if __name__ == "__main__":
    unittest.main()
