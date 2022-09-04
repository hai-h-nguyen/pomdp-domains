# -*- coding: utf-8 -*-

from curses import raw
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from general_bayes_adaptive_pomdps.domains.ordered_tool_delivery.macro_factored_all import ObjSearchDelivery_v4 as EnvToolDelivery

class OrderedToolDeliveryEnv(gym.Env):
    def __init__(self, rendering=False):

        self.core_env = EnvToolDelivery([0, 1, 2], render=rendering)

        self.action_space = spaces.Discrete(4)

        self.n_objs = 3
        self.n_human_steps = self.n_objs + 1

        self.show = rendering

        # OBSERVATION
        # discrete room locations: [2]
        # which object in the basket: [2]*n_objs
        # which object are on the table: [2]*n_objs (only observable in the tool-room)
        # human working step: [n_objs + 1] (only observable in the work-room)
        self.observation_space = spaces.MultiBinary(1 + 2*self.n_objs + self.n_human_steps)

        self.seed()

        self.coord_x_idx = 0
        self.coord_y_idx = 1
        self.timestep_idx = 2
        self.room_idx = 3

        self.max_ep_length = 20
        self.steps_taken = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _process_obs(self, raw_obs):
        if len(raw_obs) == 1:
            raw_obs = raw_obs[0]

        human_stage = raw_obs[-1]

        onehot = [0]*self.n_human_steps
        onehot[int(human_stage)] = 1

        return np.concatenate((raw_obs[:-1], onehot))

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """a known reward function
        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):
        RETURNS (`float`): the reward of the transition
        """
        # STATE
        # x_coord, y_coord
        # current primitive timestep
        # discrete room locations: [2]
        # which object in the basket: [2]*n_objs
        # which object are on the table: [2]*n_objs
        # human working step: [n_objs + 1]
        assert len(state) == (2 + 1 + 2*self.n_objs + 2)

        delta_time = new_state[self.timestep_idx] - state[self.timestep_idx]
        reward = -delta_time

        prev_human_stage = state[-1]

        new_human_stage = new_state[-1]

        # Deliver a good tool
        if prev_human_stage + 1 == new_human_stage and action == self.n_objs:
            reward += 100

        return reward

    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """True if reached end in `new_state`
        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):
        RETURNS (`bool`): whether the transition is terminal
        """
        # STATE
        # x_coord, y_coord
        # current primitive timestep
        # discrete room locations: [2]
        # which object in the basket: [2]*n_objs
        # which object are on the table: [2]*n_objs
        # human working step: [n_objs + 1]

        done = False
        human_stage = new_state[-1]

        if human_stage == 3:
            done = True

        return done

    def step(self, action):
        s = self.core_env.get_state()
        next_obs, _, _, _ = self.core_env.step([action])
        next_s = self.core_env.get_state()
        r = self.reward(s, action, next_s)
        t = self.terminal(s, action, next_s)

        if self.show:
            self.render()

        return self._process_obs(next_obs), r, t, {}

    def render(self, mode='human'):
        self.core_env.render()

    def reset(self):
        obs = self.core_env.reset()
        return self._process_obs(obs)

if __name__ == "__main__":
    env = OrderedToolDeliveryEnv()
    print(env.reset())

    print(env.step(0))
    print(env.step(3))
    print(env.step(1))
    print(env.step(3))
    print(env.step(2))
    print(env.step(3))
