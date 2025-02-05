from typing import Any, Dict

from src.control.algorithm.base import ControllerParams
from src.control.controller_factory import ControllerFactory


class JointController:
    def __init__(self, factory: ControllerFactory):
        self.factory = factory

    def build_controller(self, configs: Dict[str, ControllerParams]) -> None:
        """
        Implement the method to build the controllers for the joints

        Args:
            configs: A dictionary that maps joint names to controller parameters

        Example code:
        joint_controllers = {}
        for joint_name, config in configs.items():
            joint_controllers[joint_name] = self.factory.build(config)
        """
        pass

    def compute_control(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement the control logic here for each joint

        Input and output temporarily set as dictionaries and
        shuold be replaced with the actual upstream interface
        """
        pass

    def _fallback_control(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement the fallback control logic here for each joint

        Input and output temporarily set as dictionaries and
        shuold be replaced with the actual upstream interface
        """
        pass
