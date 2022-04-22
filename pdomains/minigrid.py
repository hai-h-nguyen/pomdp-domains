from pdomains.minigrid_core import FindGoalLightHouseTask, LightHouseEnvironment
from allenact_plugins.minigrid_plugin.minigrid_sensors import EgocentricMiniGridSensor
from allenact.base_abstractions.sensor import Sensor, SensorSuite

import gym
from gym.utils import seeding
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Sequence, cast


class MiniGridEnv(gym.Env):

    def __init__(self, world_dim: int = 1, agent_view_size: int =1, view_channels: int = 3, max_step: int = 100, **kwargs):
        self.env = LightHouseEnvironment(world_dim=world_dim, world_radius=world_radius)

        sensors = [EgocentricMiniGridSensor(
                    agent_view_size=agent_view_size,
                    view_channels=view_channels,
                )]

        self.sensors = sensors

        self.world_dim = world_dim

        self.task = FindGoalLightHouseTask(
                    env=self.env, sensors=self.sensors, task_info={}, max_steps=max_step
                )

        self.action_space = self.task.action_space
        self.observation_space = self.sensors[0]._get_observation_space()
        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32
        )

        self.steps_taken = 0
        self._found_target = False
        self.max_steps = max_step

        self.expert_view_radius = expert_view_radius

        self.num_seed: Optional[int] = None
        self.np_seeded_random_gen: Optional[np.random.RandomState] = None
        self.seed(seed=int(kwargs.get("seed", np.random.randint(0, 2 ** 31 - 1))))

        self.reset()


    def seed(self, seed: int):
        pass

    def query_expert(self):
        pass

    def query_state(self) -> np.array:
        pass

    def reset(self):
        pass

    def render(self, mode="human", **kwargs):
        pass

    def step(self, action):
        pass

    def close(self):
        pass