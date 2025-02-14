# Software Architecture Overview
## System Architecture
The repository implements a modular robotics control system with three main components:

1. Control (`/control`)
2. Planning (`/planning`)
3. Simulation environment (`/environment`)

The structure and relationships are illustrated below:
```mermaid
classDiagram
    %% Core Interfaces
    class Controller {
        <<Interface>>
        +control(state, **kwargs)
        +reset()
    }
    class Planner {
        <<Interface>>
        +plan(**kwargs)
    }
    class Env {
        <<Interface>>
        +step(action)
    }

    %% Control Module
    namespace Control {
        class ControllerFactory {
            +register_controller()
            +build()
            +build_from_dict()
        }
        class ConfigFactory {
            +register_config()
            +build()
        }
        class MLPPolicy
        class PID
        class LQR
        class Go1ControllerManager {
            +set_command()
            +switch_controller()
            +control()
        }
        class PositionController {
            +compute_command()
        }
    }

    %% Planning Module
    namespace Planning {
        class GeminiThinkingNavigator {
            +plan()
            +reset_chat()
        }
        class NavigationPlan {
            +waypoints
            +trajectory
        }
    }

    %% Environment Module
    namespace Environment {
        class Go1Env {
            +step()
            +reset()
            +render()
        }
        class TraversabilityMap {
            +load_from_grid()
            +is_valid_position()
        }
    }

    %% Relationships
    Controller <|.. MLPPolicy : implements
    Controller <|.. PID : implements
    Controller <|.. LQR : implements
    
    Planner <|.. GeminiThinkingNavigator : implements
    Env <|.. Go1Env : implements

    ControllerFactory --> Controller : creates
    ConfigFactory --> ControllerFactory : configures
    
    Go1ControllerManager --> Controller : manages
    PositionController --> Controller : uses
    
    GeminiThinkingNavigator --> NavigationPlan : produces
    Go1Env --> TraversabilityMap : uses

    %% Cross-module Interactions
    Go1ControllerManager --> Go1Env : controls
    GeminiThinkingNavigator --> TraversabilityMap : plans using
    PositionController --> Go1ControllerManager : generates commands for

    %% Design Patterns
    note for ControllerFactory "Factory Pattern"
    note for Go1ControllerManager "Strategy Pattern"
```

## Core Components
### 1. Control
Key templates:
 - [ControllerFactory](control/controller_factory.py) - Creates controller instances
 - [Controller](control/algorithm/base.py) - Control algorithm interface

Factory Pattern
- Used in control module for configuration and instantiation
- Separates object creation from algorithm logic

Strategy Pattern
- Used for switch policies during runtime (e.g. robot recovery from falling down)

### 3. Planning
- [Planner](planning/base.py) - Planning interface

### 4. Simulation Environment
- [Environment](environment/base.py) - Environment interface


## Configuration
- YAML-based configuration files in `/config`
- Separate configs for:
    - Control parameters
    - Environment settings
    - State estimation parameters
