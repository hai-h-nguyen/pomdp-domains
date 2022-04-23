
import sys

sys.path.append('/home/hainh22/Github/allenact')

from allenact_plugins.minigrid_plugin.minigrid_tasks import MiniGridTask

from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact_plugins.minigrid_plugin.minigrid_sensors import (
    EgocentricMiniGridSensor,
)

import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Sequence, cast

from allenact_plugins.minigrid_plugin.minigrid_environments import (
    FastCrossing,
)

from gym_minigrid.minigrid import Lava


class LavaCrossingS25N10Env(gym.Env):

    def __init__(self, view_channels: int = 3, agent_view_size: int = 7, max_steps: int = 100, rendering=False, **kwargs):

        grid_size = 25
        num_crossings = 10

        self.show = rendering

        env_info = {
            "size": grid_size,
            "num_crossings": num_crossings,
            "obstacle_type": Lava,
        }

        self.env = FastCrossing(**env_info)

        sensors = [EgocentricMiniGridSensor(
                    agent_view_size=agent_view_size,
                    view_channels=view_channels,
                )]

        self.sensors = sensors

        self.task = MiniGridTask(
                    env=self.env, sensors=self.sensors, task_info={}, max_steps=max_steps
                )
        self.max_steps = max_steps
        self.action_space = self.task.action_space
        self.observation_space = self.sensors[0]._get_observation_space()
        self.state_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1, ),
            dtype=np.float32
        )

        self.max_steps = max_steps

        self.num_seed: Optional[int] = None
        self.np_seeded_random_gen: Optional[np.random.RandomState] = None
        self.seed(seed=int(kwargs.get("seed", np.random.randint(0, 2 ** 31 - 1))))

        self.reset()


    def seed(self, seed: int):
        # More information about why `np_seeded_random_gen` is used rather than just `np.random.seed`
        # can be found at gym/utils/seeding.py
        # There's literature indicating that having linear correlations between seeds of multiple
        # PRNG's can correlate the outputs
        self.num_seed = seed
        self.np_seeded_random_gen, _ = cast(
            Tuple[np.random.RandomState, Any], seeding.np_random(self.num_seed)
        )

    def reset(self):
        self.env.reset()
        self.task = MiniGridTask(
            env=self.env, sensors=self.sensors, task_info={}, max_steps=self.max_steps
        )

        return self.task.get_observations()

    def query_expert(self):
        action, _ = self.task.query_expert()
        return [action]

    def query_state(self) -> np.array:
        return np.array([0])

    def step(self, action):
        assert isinstance(action, int)
        action = cast(int, action)

        _, reward, done, info = self.env.step(action)

        if self.show:
            self.render(mode='human')

        return self.task.get_observations()['minigrid_ego_image'], reward, done, info

    def close(self):
        self.env.close()

    def render(self, mode="human", **kwargs):
        self.env.render(mode=mode)