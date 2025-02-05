import jax
from mujoco_playground._src import mjx_env

from examples.mujoco_Go1.env_wrapper import Go1Env


class RecoverState:
    def __init__(self, getup_env: Go1Env, handstand_env: Go1Env):
        self.trigger_step: int = 0
        self.is_recovering: bool = False
        self._minimal_recover_steps = 50
        self.getup_env = getup_env
        self.handstand_env = handstand_env

    def check(self, state: mjx_env.State, step: int) -> bool:
        """
        Check and update the recovery state of the robot.

        Args:
            state: Current state of the environment
            step: Current simulation step

        Returns:
            bool: True if the robot is currently in recovery state
        """
        if self._needs_recovery(state):
            self._initiate_recovery(step)
        elif self._is_recovery_complete(state, step):
            self.is_recovering = False

        return self.is_recovering

    def _initiate_recovery(self, step: int) -> None:
        self.trigger_step = step
        self.is_recovering = True

    def _needs_recovery(self, state: mjx_env.State):
        return self.handstand_env.env._get_termination(state.data, state.info, jax.numpy.zeros((1)))

    def _is_recovery_complete(self, state: mjx_env.State, step: int) -> bool:
        minimal_steps_elapsed = (step - self.trigger_step) > self._minimal_recover_steps
        return minimal_steps_elapsed and self.is_upright(state) and self.is_at_desired_height(state)

    def is_upright(self, state: mjx_env.State) -> bool:
        gravity = self.getup_env.env.get_gravity(state.data)
        return self.getup_env.env._is_upright(gravity)

    def is_at_desired_height(self, state: mjx_env.State) -> bool:
        torso_height = state.data.site_xpos[self.getup_env.env._imu_site_id][2]
        return self.getup_env.env._is_at_desired_height(torso_height)
