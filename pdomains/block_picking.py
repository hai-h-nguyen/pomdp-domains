# -*- coding: utf-8 -*-

from more_itertools import first
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from helping_hands_rl_envs import env_factory
import matplotlib.pyplot as plt
import time

class BlockEnv(gym.Env):
    def __init__(self, seed=0, rendering=False, robot='kuka', action_sequence='pxyz'):

        workspace = np.asarray([[0.3, 0.7],
                                [-0.2, 0.2],
                                [0.01, 0.25]])

        self.env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 84, 'render': False, 'fast_mode': True,
                        'seed': 0, 'action_sequence': action_sequence, 'num_objects': 1, 'random_orientation': False,
                        'reward_type': 'sparse', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': robot,
                        'workspace_check': 'point', 'physics_mode': 'fast', 'hard_reset_freq': 1000, 'view_scale': 1.0,
                        'object_scale_range': (1, 1), 'obs_type': 'pixel',
                        'view_type': 'camera_center_z_height'}

        self.planner_config = {'random_orientation': True, 'dpos': 0.05, 'drot': np.pi/8}

        self.xyz_range = self.planner_config['dpos']
        self.r_range = self.planner_config['drot']

        self.env_config['render'] = rendering
        self.seed(seed)
        self.core_env = env_factory.createSingleProcessEnv('close_loop_pomdp_block_picking',
                                                            self.env_config,
                                                            self.planner_config)

        self.viewer = None
        self.show = rendering
        self.obs = None

        self.action_dim = len(self.env_config['action_sequence'])
        high_action = np.ones(self.action_dim)
        self.action_space = spaces.Box(-high_action, high_action)

        low = np.zeros((84, 84, 3))
        high = np.ones((84, 84, 3))
        self.observation_space = spaces.Box(low=low, high=high, shape=(84, 84, 3), dtype=np.float32)

        self.target_obj_idx = 0

        self.first_obs = None
        self.cnt_reset = 0

    def query_expert(self):
        action = self.core_env.getNextAction(self.target_obj_idx)
        action[1:4] /= self.xyz_range

        if self.action_dim == 5:
            action[4] /= self.r_range

        if self.env_config['robot'] == 'kuka':
            action[0] = 2*action[0] - 1

        return action

    def seed(self, seed=0):
        self.env_config['seed'] = seed
        self.np_random, seed_ = seeding.np_random(seed)
        return seed_

    @staticmethod
    def _process_obs(state, obs, reward):
        state_tile = state*np.ones((1, obs.shape[1], obs.shape[2]))
        reward_tile = reward*np.ones((1, obs.shape[1], obs.shape[2]))
        stacked = np.concatenate([obs, state_tile, reward_tile], axis=0)
        return np.transpose(stacked, (2, 1, 0))

    def step(self, action):
        action[1:4] *= self.xyz_range  # scale from [-1, 1] to [-0.05, 0.05] for xyz

        if self.action_dim == 5:  # scale from [-1, 1] to [-np.pi/8, np.pi/8]
            action[4] *= self.r_range

        if self.env_config['robot'] == 'kuka':
            action[0] = 0.5 * (action[0] + 1)  # [-1, 1] to [0, 1] for p
        (state, _, obs), reward, done = self.core_env.step(action)

        # plt.imshow(obs[0], vmin=0, vmax=0.2)
        # plt.show()

        self.obs = self._process_obs(state, obs, reward)

        info = {}

        info["success"] = done and (reward > 0)

        if self.show:
            self.render()

        return self.obs, reward, done, info

    def render(self, mode='human'):
        pass

    def reset(self):
        self.target_obj_idx = 1 - self.target_obj_idx
        self.cnt_reset += 1
        (state, _, obs) = self.core_env.reset(self.target_obj_idx)

        # plt.imshow(obs[0], vmin=0, vmax=0.2)
        # plt.show()

        # if self.cnt_reset == 2:
        #     self.first_obs = obs
        # elif self.cnt_reset > 2:
        #     print(np.min(self.first_obs - obs), np.max(self.first_obs - obs))
        self.obs = self._process_obs(state, obs, 0.0)
        return self.obs

    def close(self):
        self.core_env.close()
