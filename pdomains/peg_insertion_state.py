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

        self.rendering = rendering

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

        high_action = np.ones(3)  # delta_x, delta_y, delta_z, delta_gamma
        self.action_space = spaces.Box(-high_action, high_action)

        self.observation_space = gym.spaces.Box(
            shape=(9,), low=-np.inf, high=np.inf, dtype=np.float32
        )

        self.seed(seed)

        self.render_mode = "human"

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
        assert len(obs["all_sensors"]) == 24
        return obs["all_sensors"][:9]

    def step(self, action):
        action = self._process_action(action)
        obs, reward, done, info = self.core_env.step(action)

        info = {}

        info["success"] = reward > 0.0

        reward = self._calculate_reward(obs, action)

        if info["success"]:
            done = True

        if self.rendering:
            self.core_env.render()

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

        if self.rendering:
            self.core_env.render()

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
        obs = obs["all_sensors"]
        error_pos = obs[:3]

        reward = -error_pos[0]**2 - error_pos[1]**2 - 10*error_pos[2]**2

        d2g = np.linalg.norm(error_pos[:2])

        if d2g > 0.007:
            weight_z = 1.0
        else:
            weight_z = 0.001
        reward -= weight_z*action[2]**2

        return reward

    def close(self):
        pass
