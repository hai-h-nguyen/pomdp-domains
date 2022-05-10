# -*- coding: utf-8 -*-

import numpy as np
import gym
from gym import spaces
from pathlib import Path
from stable_baselines3 import SAC

class POMDPWrapper(gym.Wrapper):
    def __init__(self, env, partially_obs_dims: list):
        super().__init__(env)
        self.partially_obs_dims = partially_obs_dims
        # can equal to the fully-observed env
        assert 0 < len(self.partially_obs_dims) <= self.observation_space.shape[0]

        self.state_space = spaces.Box(
            low=self.observation_space.low,
            high=self.observation_space.high,
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=self.observation_space.low[self.partially_obs_dims],
            high=self.observation_space.high[self.partially_obs_dims],
            dtype=np.float32,
        )

        self.state = None

        if self.env.action_space.__class__.__name__ == "Box":
            self.act_continuous = True
            # if continuous actions, make sure in [-1, 1]
            # NOTE: policy won't use action_space.low/high, just set [-1,1]
            # this is a bad practice...
        else:
            self.act_continuous = False

    def get_obs(self, state):
        return state[self.partially_obs_dims].copy()

    def get_state(self):
        return self.state

    def reset(self):
        state = self.env.reset()  # no kwargs
        self.state = state
        return self.get_obs(state)

    def step(self, action):
        if self.act_continuous:
            # recover the action
            action = np.clip(action, -1, 1)  # first clip into [-1, 1]
            lb = self.env.action_space.low
            ub = self.env.action_space.high
            action = lb + (action + 1.0) * 0.5 * (ub - lb)
            action = np.clip(action, lb, ub)

        state, reward, done, info = self.env.step(action)
        self.state = state

        return self.get_obs(state), reward, done, info

class LunarLanderEnv(gym.Env):
    def __init__(self, seed=0, rendering=False):
        env = gym.make('LunarLanderContinuous-v2')

        partially_obs_dims=[0, 1, 4, 6, 7]

        self.core_env = POMDPWrapper(env, partially_obs_dims)

        self.viewer = None
        self.show = rendering

        self.action_space = self.core_env.action_space
        self.observation_space = self.core_env.observation_space

        expert_path = Path(__file__).resolve().parent / 'sac_lunarlander'
        self.expert = SAC.load(expert_path)

        self.seed()

    def query_expert(self):
        state = self.core_env.get_state()
        action, _ = self.expert.predict(state, deterministic=True)
        return [action]

    def seed(self, seed=None):
        self.core_env.seed(seed)

    def step(self, action):
        obs, reward, done, info = self.core_env.step(action)

        self.obs = obs

        if self.show:
            self.render()

        # no success for this task
        info["success"] = False

        return obs, reward, done, info

    def render(self, mode='human'):
        self.core_env.render()

    def reset(self):
        obs = self.core_env.reset()
        self.obs = obs
        return obs

    def close(self):
        self.core_env.close()