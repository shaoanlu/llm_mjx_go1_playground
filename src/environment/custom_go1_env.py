from typing import Any, Dict, Optional, Union

import mujoco
from etils import epath
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src.locomotion.go1.base import Go1Env as MPGo1Env
from mujoco_playground._src.locomotion.go1.base import get_assets
from mujoco_playground._src.locomotion.go1.joystick import Joystick
from mujoco_playground._src.locomotion.go1.joystick import default_config as joystick_default_config

from src.utils import download_go1_assets_from_mujoco_menagerie


class CustomGo1JoystickEnv(Joystick):
    """
    Custom Go1Joystic environment that allow using specified xml file
    """

    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict = joystick_default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super(MPGo1Env, self).__init__(config, config_overrides)

        self.download_go1_assets()

        self._mj_model = mujoco.MjModel.from_xml_string(epath.Path(xml_path).read_text(), assets=get_assets())
        self._mj_model.opt.timestep = self._config.sim_dt

        # Modify PD gains.
        self._mj_model.dof_damping[6:] = config.Kd
        self._mj_model.actuator_gainprm[:, 0] = config.Kp
        self._mj_model.actuator_biasprm[:, 1] = -config.Kp

        # Increase offscreen framebuffer size to render at higher resolutions.
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = xml_path
        self._imu_site_id = self._mj_model.site("imu").id

        self.env_cfg = self._config
        self._post_init()

    def download_go1_assets(self):
        download_go1_assets_from_mujoco_menagerie()
