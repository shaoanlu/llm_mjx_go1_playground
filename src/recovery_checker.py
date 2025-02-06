import jax
from mujoco import mjx
from mujoco_playground._src import mjx_env

from src.environment.env_wrapper import Go1Env


UPVECTOR_SENSOR = "upvector"


class RecoverState:
    def __init__(self, env: Go1Env):
        self.trigger_step: int = 0
        self.is_recovering: bool = False
        self._minimal_recover_steps = 50
        self.env = env

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
        return self._get_termination(state.data)

    def _is_recovery_complete(self, state: mjx_env.State, step: int) -> bool:
        minimal_steps_elapsed = (step - self.trigger_step) > self._minimal_recover_steps
        return minimal_steps_elapsed and self.is_upright(state) and self.is_at_desired_height(state)

    def is_upright(self, state: mjx_env.State) -> bool:
        gravity = self._get_gravity(state.data)
        return self._is_upright(gravity)

    def is_at_desired_height(
        self, state: mjx_env.State, pos_tol: float = 0.005, z_des: float = 0.275
    ) -> bool:
        torso_height = state.data.site_xpos[self.env.env._imu_site_id][2]
        height = jax.numpy.min(jax.numpy.array([torso_height, z_des]))
        height_error = z_des - height
        return height_error < pos_tol

    def _get_termination(self, data: mjx.Data):
        fall_termination = self._get_upvector(data)[-1] < -0.25
        energy = jax.numpy.sum(jax.numpy.abs(data.actuator_force) * jax.numpy.abs(data.qvel[6:]))
        return fall_termination

    def _get_upvector(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.env.env.mj_model, data, UPVECTOR_SENSOR)

    def _get_gravity(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self.env.env._imu_site_id].T @ jax.numpy.array([0, 0, -1])

    def _is_upright(self, gravity: jax.Array, ori_tol: float = 0.01) -> jax.Array:
        ori_error = jax.numpy.sum(jax.numpy.square(jax.numpy.array([0.0, 0.0, -1.0]) - gravity))
        return ori_error < ori_tol
