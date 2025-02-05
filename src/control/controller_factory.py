from typing import Any, Dict, Type

from src.control.algorithm.base import Controller, ControllerParams, ControllerParamsBuilder
from src.control.algorithm.mpc import MPC, MPCParams, MPCParamsBuilder
from src.control.algorithm.pid import PID, PIDParams, PIDParamsBuilder


class ConfigFactory:
    """
    This class build controller parameters from a configuration dictionary (which is loaded from yaml files).
    """

    def __init__(self):
        # register parameter builders
        self.params_builder_map: Dict[str, Type[ControllerParams]] = {
            "mpc": MPCParamsBuilder,
            "pid": PIDParamsBuilder,
        }

    def register_config(self, key: str, value: Type[ControllerParams]) -> None:
        self.params_builder_map[key] = value

    def build(self, config: Dict[str, Any]) -> ControllerParams:
        algorithm_type: str = config.get("algorithm_type", "algorithm_type_not_defined").lower()
        params_builder: Type[ControllerParamsBuilder] | None = self.params_builder_map.get(algorithm_type)
        if params_builder is None:
            raise ValueError(
                f"Invalid algorithm type: {algorithm_type}. Valid types are: {list(self.params_builder_map.keys())}"
                "\n config: {config}"
            )
        else:
            return params_builder().build(config)


class ControllerFactory:
    """
    This class builds a controller based on the controller parameters (which is built by `ConfigFactory`).

    Example usage:
        config = load_yaml(path_to_config_file)
        controller_params = ConfigFactory().build(config)
        controller = ControllerFactory().build(controller_params)
    """

    def __init__(self):
        # register controller classes
        self.controller_map: Dict[Type[ControllerParams], Type[Controller]] = {
            MPCParams: MPC,
            PIDParams: PID,
        }
        self.config_factory: ConfigFactory | None = None

    def register_controller(self, key: Type[ControllerParams], value: Type[Controller]):
        self.controller_map[key] = value

    def build(self, params: ControllerParams) -> Controller:
        """Build controller from controller parameters."""
        controller_class: Type[Controller] | None = self.controller_map.get(type(params))
        if controller_class is None:
            raise ValueError(
                f"Unsupported parameter type: {params.__class__.__name__}. "
                f"Supported types are: {[cls.__name__ for cls in self.controller_map.keys()]}"
            )
        return controller_class(params)

    def build_from_dict(self, params: Dict[str, Any]) -> Controller:
        """
        Build controller from configuration dictionary by first building controller parameters.
        Use this method when the dict is simple and its definition does not affect code readability.

        NOTE: ther will be errors if the configuration is not registered in the ConfigFactory.
        """
        self.config_factory = ConfigFactory() if not self.config_factory else self.config_factory
        controller_params: ControllerParams = self.config_factory.build(params)
        return self.build(controller_params)
