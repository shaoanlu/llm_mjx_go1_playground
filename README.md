# MJX go1 playground with LLM navigation

This repository provides a toy example of using an LLM (Gemini 2.0 Thinking) to navigate a quadruped robot in an unknown environment through trial and error.

The LLM is given only the map size (5x5). Throughout the iterations, it receives feedback on the success or failure of the robot's attempts to follow waypoints, along with information about the traversed cells. This allows the LLM to potentially refine its guesses over time.
![](assets/llm_go1_navigation.png)

## Requirements
- Python 3.10+
- numpy
- scipy
- yaml
- mujoco_playground
- jax
- mujoco


## Result
### 1. Go1 locomotion
The model was trained using notebooks in [mujoco_playground](https://github.com/google-deepmind/mujoco_playground)

![](examples/gifs/ppo_Go1JoystickFlatTerrain.gif) ![](examples/gifs/ppo_Go1Handstand_Go1Getup_Go1Joystick_Go1Footstand.gif)

### 2. LLM navigation via trial-and-error
Leveraging the trained locomotion policy, this demo further enhance its autonomy by introducing navigation feature using LLM. Prototype of navigation of LLM w/o the quadruped simulation can be found in [this repo](https://github.com/shaoanlu/gemini_maze_exploration).

![](examples/gifs/go1_LLM_Navigation.gif)

## Execution
### 1. Go1 locomotion
Refer to [`colab demo`](examples/locomotion.ipynb) notebook or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaoanlu/llm_mjx_go1_playground/blob/main/examples/locomotion.ipynb)

### 2. LLM navigation via trial-and-error
Refer to [`colab demo`](llm_navigation.ipynb) notebook or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaoanlu/llm_mjx_go1_playground/blob/main/examples/llm_navigation.ipynb)

## Memo
- `ruff format`
- `ruff check` or `ruff check --fix`