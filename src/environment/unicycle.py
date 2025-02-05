from src.environment.base import Env


class UnicycleEnv(Env):
    def __init__(self, config):
        self.config = config

    def step(self, action):
        pass

    def reset(self):
        pass
