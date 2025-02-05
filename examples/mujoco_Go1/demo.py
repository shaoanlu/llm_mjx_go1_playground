"""
This demo script is modified from the locomotion tutorial of mujoco_playground
https://github.com/google-deepmind/mujoco_playground
"""

import argparse

import jax
import mediapy
import mujoco
from tqdm import tqdm

from examples.mujoco_Go1.env_wrapper import Go1Env
from examples.mujoco_Go1.ppo import PPO, PPOParams, PPOParamsBuilder
from src.control.controller_factory import ControllerFactory


parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="Go1Handstand", choices=["Go1Handstand", "Go1JoystickFlatTerrain"])
args = parser.parse_args()


def main():
    env_name = args.env_name

    # Instantiate simulator
    rng = jax.random.PRNGKey(12345)
    env = Go1Env(env_name=env_name)
    state = env.reset(rng)

    # Instantiate controller
    factory = ControllerFactory()
    factory.register_controller(PPOParams, PPO)
    controller_config = {"npy_path": f"examples/mujoco_Go1/nn_params/{env_name}"}
    ppo_params = PPOParamsBuilder().build(config=controller_config)
    controller = factory.build(params=ppo_params)

    # start closed-loop simulation
    rollout = []
    actions = []
    cmd_x = 0
    for i in tqdm(range(env.env_cfg.episode_length)):
        if (env_name == "Go1JoystickFlatTerrain") and (i % 150 == 0):
            # Increase the forward velocity by 0.25 m/s every 150 steps.
            cmd_x += 0.25
            command = jax.numpy.array([cmd_x, 0, 0])
            state.info["command"] = command
        ctrl = controller.control(state.obs["state"])  # do not use privileged_state
        state = env.step(state, ctrl)

        # record result
        actions.append(ctrl)
        rollout.append(state)

    # visualize simulation result
    render_every = 2
    fps = 1.0 / env.dt / render_every
    traj = rollout[::render_every]

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

    frames = env.render(
        traj,
        camera="side" if env_name == "Go1Handstand" else "track",
        scene_option=scene_option,
        height=480,
        width=640,
    )
    mediapy.show_video(frames, fps=fps, loop=False)


if __name__ == "__main__":
    main()
