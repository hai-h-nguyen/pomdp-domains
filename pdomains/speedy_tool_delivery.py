# -*- coding: utf-8 -*-

from curses import raw
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from general_bayes_adaptive_pomdps.domains.speedy_tool_delivery.macro_factored_all import ObjSearchDelivery_v4 as EnvToolDelivery

class SpeedyToolDeliveryEnv(gym.Env):
    def __init__(self, rendering=False):

        self.human_speeds = [30, 10]
        self.core_env = EnvToolDelivery(human_speeds=self.human_speeds, render=rendering)

        self.action_space = spaces.Discrete(5)

        self.show = rendering

        self.n_objs = 3
        self.n_human_steps = self.n_objs + 1

        # OBSERVATION
        # discrete room locations: [3]
        # which object in the basket: [3]*n_objs
        # which object are on the table: [2]*n_objs (only observable in the tool-room)
        # human 0 working step: [n_objs + 1] (only observable in the work-room)
        # human 1 working step: [n_objs + 1] (only observable in the work-room)
        self.observation_space = spaces.MultiBinary(3 + 2*self.n_objs + self.n_objs +
                                                    self.n_human_steps + self.n_human_steps)

        self.seed()

        self.coord_x_idx = 0
        self.coord_y_idx = 1
        self.timestep_idx = 2
        self.room_idx = 3

        self.max_ep_length = 20
        self.steps_taken = 0

        self.discount = 0.95

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _to_onehot(self, val, size):
        temp = [0]*size
        temp[int(val)] = 1
        return temp

    def _process_obs(self, raw_obs):
        # OBSERVATION
        # discrete room locations: [3]
        # which object in the basket: [3]*n_objs
        # which object are on the table: [2]*n_objs (only observable in the tool-room)
        # human 0 working step: [n_objs + 1] (only observable in the work-room)
        # human 1 working step: [n_objs + 1] (only observable in the work-room)
        if len(raw_obs) == 1:
            raw_obs = raw_obs[0]

        assert len(raw_obs) == 9, len(raw_obs)

        room_location = raw_obs[0]
        onehot_room = self._to_onehot(room_location, 3)

        obj0_in_basket = raw_obs[1]
        obj0_onehot = self._to_onehot(obj0_in_basket, 3)

        obj1_in_basket = raw_obs[2]
        obj1_onehot = self._to_onehot(obj1_in_basket, 3)

        obj2_in_basket = raw_obs[3]
        obj2_onehot = self._to_onehot(obj2_in_basket, 3)

        objs_in_table = raw_obs[4:7]

        human0_stage = raw_obs[7]
        onehot_human0_stage = self._to_onehot(human0_stage, 2)

        human1_stage = raw_obs[8]
        onehot_human1_stage = self._to_onehot(human1_stage, 2)

        return np.array(onehot_room + obj0_onehot + obj1_onehot + obj2_onehot
                        + list(objs_in_table) + onehot_human0_stage + onehot_human1_stage)

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """a known reward function

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`float`): the reward of the transition

        """
        # STATE
        # x_coord, y_coord [2] [2]
        # current primitive timestep [max_step]
        # discrete room locations: [3]
        # which object in the basket: [3]*n_objs
        # which object are on the table: [2]*n_objs
        # human 0 working step: [n_objs + 1]
        # human 1 working step: [n_objs + 1]
        # human 0 is working or not [2]
        # human 1 is working or not [2]
        assert len(state) == (2 + 1 + 1 + 2*self.n_objs + 2 + 2)

        delta_time = new_state[self.timestep_idx] - state[self.timestep_idx]
        reward = -delta_time

        if self.human_speeds[0] < self.human_speeds[1]:
            human0_wait_penalty = -5
            human1_wait_penalty = -5
        else:
            human0_wait_penalty = -5
            human1_wait_penalty = -5

        # human 0
        prev_human0_stage = state[-4]
        new_human0_stage = new_state[-4]
        human0_waiting = state[-2]

        # Deliver a good tool for human 0
        if prev_human0_stage + 1 == new_human0_stage and action == self.n_objs:
            reward += 100

        if human0_waiting:
            reward += human0_wait_penalty

        # human 1
        prev_human1_stage = state[-3]
        new_human1_stage = new_state[-3]
        human1_waiting = state[-1]

        if human1_waiting:
            reward += human1_wait_penalty

        # Deliver a good tool for human 1
        if prev_human1_stage + 1 == new_human1_stage and action == self.n_objs + 1:
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
        # x_coord, y_coord [2] [2]
        # current primitive timestep [max_step]
        # discrete room locations: [3]
        # which object in the basket: [3]*n_objs
        # which object are on the table: [2]*n_objs
        # human 0 working step: [n_objs + 1]
        # human 1 working step: [n_objs + 1]
        # human 0 is waiting or not [2]
        # human 1 is waiting or not [2]

        done = False
        human_stage_0 = new_state[-4]
        human_stage_1 = new_state[-3]

        if human_stage_0 >= self.n_objs and human_stage_1 >= self.n_objs:
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
    env = SpeedyToolDeliveryEnv()
    print(env.reset())

    print(env.step(0))
    print(env.step(3))
    print(env.step(1))
    print(env.step(3))
    print(env.step(2))
    print(env.step(3))
