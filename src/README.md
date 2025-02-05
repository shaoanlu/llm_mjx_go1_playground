# Software Architecture Overview
## System Architecture
The repository implements a modular robotics control system with four main components:

1. Control (`/control`)
2. Estimation (`/estimation`)
3. Planning (`/planning`)
4. Simulation environment (`/env`)

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

    subgraph State Estimation
        SE[State Estimator]
        BF[Base Filter]
        KF[Kalman Filter]
        PF[Particle Filter]
        CF2[Complementary Filter]
        SE --> BF
        BF --> KF
        BF --> PF
        BF --> CF2
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
    Estimation --> Control
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
  
### 2. Estimation
Key templates:
  - [StateEstimator](estimation/state_estimator.py) - Creates state estimator interface consisting of fileter algorithms
  - [BaseFilter](estimation/algorithm/base.py) - Filter algorithm interface

Observer pattern
- Implemented in state estimation for sensor fusion
- Allows multiple filters to update state independently

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
