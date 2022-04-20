from pdomains.lighthouse_core import FindGoalLightHouseTask, LightHouseEnvironment
from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact_plugins.lighthouse_plugin.lighthouse_sensors import (
    FactorialDesignCornerSensor,
)

import gym
from gym.utils import seeding
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Sequence, cast


DISCOUNT_FACTOR = 0.99
STEP_PENALTY = -0.01
FOUND_TARGET_REWARD = 1.0

class LightHouseEnv(gym.Env):

    def __init__(self, world_dim: int = 1, world_radius: int = 15, max_step: int = 100, **kwargs):
        self.env = LightHouseEnvironment(world_dim=world_dim, world_radius=world_radius)

        sensors = [
            FactorialDesignCornerSensor(
                view_radius=world_radius,
                world_dim=world_dim,
                degree=-1,
            )
        ]

        self.sensors = (
                    SensorSuite(sensors) if not isinstance(sensors, SensorSuite) else sensors
                )


        self.task = FindGoalLightHouseTask(
                    env=self.env, sensors=self.sensors, task_info={}, max_steps=max_step
                )

        self.action_space = self.task.action_space

        self.steps_taken = 0
        self._found_target = False
        self.max_steps = max_step

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
        self.steps_taken = 0
        self._found_target = False

        self.env.random_reset()

        return self.task.get_observations()


    def step(self, action):
        assert isinstance(action, int)
        action = cast(int, action)

        self.steps_taken += 1

        done = False

        self.env.step(action)
        reward = STEP_PENALTY

        if np.all(self.env.current_position == self.env.goal_position):
            self._found_target = True
            reward += FOUND_TARGET_REWARD
            done = True
        elif self.steps_taken == self.max_steps - 1:
            reward = STEP_PENALTY / (1 - DISCOUNT_FACTOR)
            done = True

        return self.task.get_observations()['corner_fixed_radius_categorical'], reward, done, {}

    def close(self):
        pass

env = LightHouseEnv()
env.reset()
print(env.step(0))