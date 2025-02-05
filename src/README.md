# Software Architecture Overview
## System Architecture
The repository implements a modular robotics control system with four main components:

1. Control (`/control`)
2. Planning (`/planning`)
3. Simulation environment (`/env`)

```mermaid
graph TB
    subgraph Core Components
        Control[Control System]
        Estimation[State Estimation]
        Planning[Path Planning]
        Env[Simulation Environment]
    end

    subgraph Control System
        CF[Controller Factory]
        BC[Base Controller]
        MPC[MPC Controller]
        PID[PID Controller]
        CF --> BC
        BC --> MPC
        BC --> PID
    end

    subgraph Path Planning
        BP[Base Planner]
        RRT[RRT Planner]
        BP --> RRT
    end

    subgraph Configuration
        Config[YAML Configs]
        Config --> |Params| Control
        Config --> |Params| Estimation 
        Config --> |Params| Planning
        Config --> |Params| Env
    end

    Control --> Env
    Planning --> Control
    Env --> Estimation
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
- To be implemented. Enables runtime algorithm selection

### 3. Planning
- TBU

### 4. Simulation Environment
- TBU


## Configuration
- YAML-based configuration files in `/config`
- Separate configs for:
    - Control parameters
    - Environment settings
    - State estimation parameters
