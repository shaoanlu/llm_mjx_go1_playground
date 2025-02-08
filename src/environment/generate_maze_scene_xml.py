"""
This module provides functionality to generate XML descriptions of maze environments
for the MuJoCo physics simulator. It creates environments with walls, obstacles,
and goal positions suitable for robotics navigation tasks.
"""

from typing import Tuple

import numpy as np

from src.utils import generate_maze


BASE_ENV_XML = """<mujoco model="go1 scene">
  <include file="go1_mjx.xml"/>

  <statistic center="0 0 0.1" extent="0.8" meansize="0.04"/>

  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
    <rgba force="1 0 0 1"/>
    <global azimuth="120" elevation="-20"/>
    <map force="0.01"/>
    <scale forcewidth="0.3" contactwidth="0.5" contactheight="0.2"/>
    <quality shadowsize="8192"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0"/>
    <material name="box_material" shininess="0.1" specular="0.4" roughness="0.3" />
  </asset>

  <worldbody>
    <geom name="floor" size="0 0 0.01" type="plane" material="groundplane" priority="1" friction="0.6" condim="3" contype="1" conaffinity="0"/>

"""

BOUNDARY_AND_GOAL_XML = """
    <body name="outer_wall_body1" pos="2.5 5.5 0.5">
      <geom name="outer_wall1" type="box" size="3.5 0.5 0.5" material="box_material" rgba="0.0 0.72 0.53 1.0" contype="0" conaffinity="0"/>
    </body>
    <body name="outer_wall_body2" pos="5.5 2.5 0.5">
      <geom name="outer_wall2" type="box" size="0.5 3.5 0.5" material="box_material" rgba="0.0 0.72 0.53 1.0" contype="0" conaffinity="0"/>
    </body>
    <body name="outer_wall_body3" pos="-0.5 3 0.5">
      <geom name="outer_wall3" type="box" size="0.5 2 0.5" material="box_material" rgba="0.0 0.72 0.53 1.0" contype="0" conaffinity="0"/>
    </body>
    <body name="outer_wall_body4" pos="3.0 -0.5 0.5">
      <geom name="outer_wall4" type="box" size="2 0.5 0.5" material="box_material" rgba="0.0 0.72 0.53 1.0" contype="0" conaffinity="0"/>
    </body>
    <body name="goal_box_body" pos="4.5 4.5 0.01">
      <geom name="goal_box" type="box" size="0.5 0.5 0.01" material="box_material" rgba="0.7 0.2 0.3 1.0" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <keyframe>
    <key name="home" qpos="
    0 0 0.278
    1 0 0 0
    0.1 0.9 -1.8
    -0.1 0.9 -1.8
    0.1 0.9 -1.8
    -0.1 0.9 -1.8"
      ctrl="0.1 0.9 -1.8 -0.1 0.9 -1.8 0.1 0.9 -1.8 -0.1 0.9 -1.8"/>
    <key name="home_higher" qpos="0 0 0.31 1 0 0 0 0 0.82 -1.63 0 0.82 -1.63 0 0.82 -1.63 0 0.82 -1.63"
      ctrl="0 0.82 -1.63 0 0.82 -1.63 0 0.82 -1.63 0 0.82 -1.63"/>
    <key name="pre_recovery"
      qpos="-0.0318481 -0.000215369 0.0579031 1 -2.70738e-05 6.06169e-05 0.000231261 -0.352275 1.18554 -2.80738 0.360892 1.1806 -2.80281 -0.381197 1.16812 -2.79123 0.391054 1.1622 -2.78576"
      ctrl="-0.352275 1.18554 -2.80738 0.360892 1.1806 -2.80281 -0.381197 1.16812 -2.79123 0.391054 1.1622 -2.78576"/>
    <key name="footstand"
      qpos="0 0 0.54 0.8 0 -0.8 0 0 0.82 -1.6 0 0.82 -1.68 0 1.82 -1.16 0.0 1.82 -1.16"
      ctrl="0 0.82 -1.6 0 0.82 -1.68 0 1.82 -1.16 0.0 1.82 -1.16"/>
    <key name="handstand"
      qpos="0 0 0.54 0.8 0 0.8 0 0 -0.686 -1.16 0 -0.686 -1.16 0 1.7 -1.853 0 1.7 -1.853"
      ctrl="0 -0.686 -1.16 0 -0.686 -1.16 0 1.7 -1.853 0 1.7 -1.853"/>
  </keyframe>
</mujoco>
"""

INTERIOR_WALLS_TEMPLATE = """
    [ADD_WALL_BLOCKS_HERE]
"""


def generate_wall_body_geom(x: int, y: int, wall_id: int) -> str:
    """
    Generate XML string for a single maze wall block at specified grid coordinates.

    Args:
        x (int): X-coordinate of the wall in the maze grid
        y (int): Y-coordinate of the wall in the maze grid
        wall_id (int): Unique identifier for the wall, used in naming

    Returns:
        str: XML string defining a box geom with specified position and properties
    """
    block_center_x = x + 0.5
    block_center_y = y + 0.5
    wall_height = 0.5

    return f"""<body name=\"box_body{wall_id}\" pos=\"{block_center_x} {block_center_y} {wall_height}\">
      <geom name="static_box{wall_id}" type="box" size="0.5 0.5 0.5" material="box_material" rgba="0.0 0.72 0.53 1.0" contype="0" conaffinity="0"/>
    </body>
    """


def generate_maze_xml(
    height: int = 3,
    width: int = 3,
    base_env_xml: str = BASE_ENV_XML,
    boundary_goal_xml: str = BOUNDARY_AND_GOAL_XML,
    interior_walls_template: str = INTERIOR_WALLS_TEMPLATE,
) -> Tuple[np.ndarray, str]:
    """
    Generate a complete XML description of a maze environment for MuJoCo.

    This function creates a maze environment by:
    1. Generating a random maze using the generate_maze utility
    2. Converting the maze to a grid of walls
    3. Creating XML descriptions for each wall
    4. Combining wall descriptions with base environment templates

    Args:
        height (int, optional): Height of the maze in grid cells. Defaults to 3.
        width (int, optional): Width of the maze in grid cells. Defaults to 3.
        base_env_xml (str, optional): Template for environment header. Defaults to BASE_ENV_XML.
        boundary_goal_xml (str, optional): Template for environment footer. Defaults to BOUNDARY_AND_GOAL_XML.
        interior_walls_template (str, optional): Template for wall placement. Defaults to INTERIOR_WALLS_TEMPLATE.

    Returns:
        np.ndarray: the generate maze
        str: Complete XML string describing the maze environment
    """
    # Generate random maze layout
    maze = generate_maze(height, width)
    grid = np.array(maze).astype(np.int32)

    # Remove outer walls as they're handled by boundary_goal_xml
    grid = grid[1:-1, 1:-1]  # (7x7) -> (5x5) by removing outer walls

    interior_walls_xml = ""
    cube_count = 0
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 0:  # 0 indicates wall presence
                cube_count += 1
                interior_walls_xml += generate_wall_body_geom(x, y, cube_count)

    # Combine all XML components
    return (
        grid,
        base_env_xml
        + interior_walls_template.replace("[ADD_WALL_BLOCKS_HERE]", interior_walls_xml)
        + boundary_goal_xml,
    )
