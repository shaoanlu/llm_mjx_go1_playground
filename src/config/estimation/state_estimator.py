"""
Demonstrate an alternative way to define configurations other than ParamsBuilder

Reference:
- https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/config/
"""

from typing import Any, Dict


def get_state_estimator_config(env_name: str) -> Dict[str, Any]:
    """
    Get the state estimator configuration for the environment

    Args:
        env_name: Name of the environment

    Returns:
        The configuration for the state estimator
    """
    if env_name == "simple":
        return {
            "filters": {
                "kalman_filter": {"process_noise": [0.1, 0.1], "measurement_noise": [0.1, 0.1]},
            },
            "initial_state": {"x": [0.0, 0.0], "xd": [0.0, 0.0]},
            "sampling_time": 0.01,
        }
    elif env_name == "complex":
        return {
            "filters": {
                "kalman_filter": {
                    "sensor_type": "imu",
                    "parameters": {"process_noise": [0.1, 0.1, 0.01], "measurement_noise": [0.1, 0.1, 0.01]},
                },
                "particle_filter": {"sensor_type": "joint", "parameters": {"num_particles": 100}},
            },
            "initial_state": {"x": [0.0, 0.0], "xd": [0.0, 0.0]},
            "sampling_time": 0.01,
        }
    else:
        raise ValueError(f"Unknown environment: {env_name}")
