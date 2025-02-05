# Adopting The Template to Make A New PPO Policy For Mujoco Go1 Tasks

## File Description
- `ppo.py`: Implement a MLP network as well as the `Controller` interfaces based on the repo template.
- `env_wrapper.py` A wrapper to mujoco env to the `Env` interface of the repo template.
- `recovery_checker.py` Implement a falldown checker for the robot. Used for triggering the recovery (`Getup`) strategy.
- `demo.py`: The demo script. Result is shown below.
- `colab_demo.ipynb`: Another demosacript, which show more usage of design patterns (Factory, Strategy, and Adapter) in the context of Go1 control.

## Result
![](gifs/ppo_Go1JoystickFlatTerrain.gif) ![](gifs/ppo_Go1Handstand_Go1Getup_Go1Joystick_Go1Footstand.gif)

## Installation
```bash
pip install mujoco mujoco_mjx brax playground mediapy
```

## Execution
### In colab (recommended)
See [`colab_demo`](colab_demo.ipynb) notebook or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaoanlu/control_system_project_template/blob/main/examples/mujoco_Go1/colab_demo.ipynb)

### Local
```bash
# navigate to root folder of the repo
python3 examples/mujoco_Go1/demo.py  --env_name Go1Handstand
# or
python3 examples/mujoco_Go1/demo.py  --env_name Go1JoystickFlatTerrain
```

## Architecture
```mermaid
---
title: Go1 PPO Controller Class Hierarchy
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

    class PPOParams {
        +nn_num_layers: int
        +nn_params: Dict
        +algorithm_type: str
    }

    class PPOParamsBuilder {
        +build(config: Dict) PPOParams
    }

    class PPO {
        -params: PPOParams
        -_inference: Callable
        +__init__(params: PPOParams)
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

    class PPOJoystick2HandstandAdapter {
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

    Controller <|-- PPOJoystick2HandstandAdapter
    Controller <|-- PPO
    PPOJoystick2HandstandAdapter o-- Controller
    Go1ControllerManager o-- "1..*" Controller
    Go1ControllerManager o-- Go1ControllerType
    ControllerParams <|-- PPOParams
    ControllerParamsBuilder <|-- PPOParamsBuilder
    PPOParamsBuilder ..> PPOParams : creates
    PPO o-- PPOParams
```


### Learning notes
- Each Go robot task uses a different environmental configuration (including noise parameters, mojoco model, initial poses, randomization settings, actuation calculations, etc.). Policies trained for one task generally don't perform well when applied to different task environments.
  - I was a little bit surprised as I expect RL policies be more robust.
  - Joystick XML: `FEET_ONLY_FLAT_TERRAIN_XML` (only feet-ground collision is enabled, also diff trunk collision geometry)
  - Handstand XML: `FULL_FLAT_TERRAIN_XML`