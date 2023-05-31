# -*- coding: utf-8 -*-

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper


class PegInsertionEnv(gym.Env):
    def __init__(self, rendering=False, seed=0):

        # Get controller config
        controller_config = load_controller_config(default_controller="IK_POSE")

        # Create argument configuration
        config = {
            "env_name": "SoftPegInHole",
            "robots": "SoftUR5e",
            "controller_configs": controller_config,
        }

        # Create environment
        env = suite.make(
            **config,
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera="agentview",
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=10,
            hard_reset=False,
        )

        # Wrap this environment in a visualization wrapper
        self.core_env = VisualizationWrapper(env, indicator_configs=None)

        high_action = np.ones(4)  # delta_x, delta_y, delta_z, delta_gamma
        self.action_space = spaces.Box(-high_action, high_action)

        self.observation_space = gym.spaces.Box(
            shape=(9,), low=-np.inf, high=np.inf, dtype=np.float32
        )

        self.seed(seed=seed)

    def query_expert(self):
        """_summary_

        Args:

        Returns:
            _type_: expert action
        """
        pass

    def seed(self, seed=0):
        self.np_random, seed_ = seeding.np_random(seed)
        return seed_

    def _process_obs(self, obs):
        """
        select features to create the observation
        """
        all_data = obs["all_sensors"]
        return all_data[-9:]

    def step(self, action):
        action = self._process_action(action)
        obs, reward, done, info = self.core_env.step(action)

        info = {}

        info["success"] = reward > 0.0

        return self._process_obs(obs), reward, done, info

    def render(self, mode='human'):
        self.core_env.render()

    def reset(self):
        """
        randomize the initial position of the peg
        """
        self.core_env.reset()

        action = self.np_random.uniform(-1, 1, size=6)

        action = self._process_action(action)

        obs, _, _, _ = self.core_env.step(action)

        return self._process_obs(obs)

    def _process_action(self, action):
        """
        zero out the gripper action and the rotations along XY axes
        """
        sent_action = np.zeros(7)
        sent_action[-1] = -1  # gripper
        sent_action[:3] = action[:3]  # delta x, y, z
        # sent_action[5] = action[3]  # delta gamma

        return sent_action*0.025

    def _calculate_reward(self, obs, action):
        """
        calculate dense reward for training SAC w. state
        """
        pass

    def close(self):
        pass
