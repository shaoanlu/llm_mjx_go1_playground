# Adopting The Template to Make A New MLPPolicy Policy For Mujoco Go1 Tasks


## Result
![](gifs/ppo_Go1JoystickFlatTerrain.gif) ![](gifs/ppo_Go1Handstand_Go1Getup_Go1Joystick_Go1Footstand.gif)

## Installation
```bash
pip install mujoco mujoco_mjx brax playground mediapy
```

## Execution
### In colab (recommended)
See [`colab demo`](locomotion.ipynb) notebook or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaoanlu/llm_mjx_playground/blob/main/examples/colab_demo.ipynb)


## Architecture
```mermaid
---
title: Go1 MLPPolicy Controller Class Hierarchy
---
classDiagram
    class Controller {
        <<abstract>>
        +control(state, command?, data?) np.ndarray
    }

    class ControllerParams {
        <<abstract>>
    }

    class ControllerParamsBuilder {
        <<abstract>>
        +build(config: Dict)*
    }

    class MLPPolicy {
        +nn_num_layers: int
        +nn_params: Dict
        +algorithm_type: str
    }

    class MLPPolicyBuilder {
        +build(config: Dict) MLPPolicy
    }

    class MLPPolicy {
        -params: MLPPolicy
        -_inference: Callable
        +__init__(params: MLPPolicy)
        +control(state) np.ndarray
        +build_network()
        -forward(x: np.ndarray)
    }

    class Go1ControllerType {
        <<enumeration>>
        JOYSTICK
        HANDSTAND
        FOOTSTAND
    }

    class MLPPolicyJoystick2HandstandAdapter {
        -_controller: Controller
        -_src_env: Go1Env
        -_tar_env: Go1Env
        +control(state, command, data) np.ndarray
    }

    class Go1ControllerManager {
        -_controllers: Dict[Go1ControllerType, Controller]
        -_active_type: Go1ControllerType
        -_command: np.ndarray
        +set_command(command)
        +switch_controller(controller_type)
        +control(state) np.ndarray
    }

    Controller <|-- MLPPolicyJoystick2HandstandAdapter
    Controller <|-- MLPPolicy
    MLPPolicyJoystick2HandstandAdapter o-- Controller
    Go1ControllerManager o-- "1..*" Controller
    Go1ControllerManager o-- Go1ControllerType
    ControllerParams <|-- MLPPolicy
    ControllerParamsBuilder <|-- MLPPolicyBuilder
    MLPPolicyBuilder ..> MLPPolicy : creates
    MLPPolicy o-- MLPPolicyParams
```


### Learning notes
- Each Go robot task uses a different environmental configuration (including noise parameters, mojoco model, initial poses, randomization settings, actuation calculations, etc.). Policies trained for one task generally don't perform well when applied to different task environments.
  - I was a little bit surprised as I expect RL policies be more robust.
  - Joystick XML: `FEET_ONLY_FLAT_TERRAIN_XML` (only feet-ground collision is enabled, also diff trunk collision geometry)
  - Handstand XML: `FULL_FLAT_TERRAIN_XML`