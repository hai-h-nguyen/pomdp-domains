# -*- coding: utf-8 -*-

from curses import raw
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from general_bayes_adaptive_pomdps.domains.speedy_tool_delivery.macro_factored_all import ObjSearchDelivery_v4 as EnvToolDelivery

class SpeedyToolDeliveryEnv(gym.Env):
    def __init__(self, human_speeds=[20, 10], rendering=False):

        self.human_speeds = human_speeds
        self.core_env = EnvToolDelivery(human_speeds=self.human_speeds, render=rendering)

        self.action_space = spaces.Discrete(5)

        self.show = rendering

        self.n_objs = 3
        self.n_human_steps = self.n_objs + 1

        # OBSERVATION
        # discrete room locations: [3]
        # which object in the basket: [3]*n_objs
        # which object are on the table: [2]*n_objs (only observable in the tool-room)
        # human 0 working step: [n_human_steps] (only observable in the work-room)
        # human 1 working step: [n_human_steps] (only observable in the work-room)
        # it is really one-hot encoded
        self.observation_space = spaces.MultiBinary(3 + 3*self.n_objs + self.n_objs +
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
        onehot_human0_stage = self._to_onehot(human0_stage, 4)

        human1_stage = raw_obs[8]
        onehot_human1_stage = self._to_onehot(human1_stage, 4)

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
        # which object is waited for - human 0: [n_objs]
        # which object is waited for - human 1: [n_objs]
        assert len(state) == (2*self.n_objs + 6), f"Len state {len(state)} is wrong"

        delta_time = new_state[self.timestep_idx] - state[self.timestep_idx]
        reward = -delta_time

        # faster human waiting incurs more penalty
        if self.human_speeds[0] < self.human_speeds[1]:
            human0_wait_penalty = -30
            human1_wait_penalty = -10
        else:
            human0_wait_penalty = -10
            human1_wait_penalty = -30

        objs_in_basket = state[4:4+self.n_objs]
        next_objs_in_basket = new_state[4:4+self.n_objs]

        # human 0
        curr_human0_needed_obj = int(state[-2])

        deliver_success = (action == self.n_objs) and \
                          (sum(next_objs_in_basket) < sum(objs_in_basket))

        # Deliver a good tool for human 0
        if objs_in_basket[curr_human0_needed_obj] > 0 and deliver_success:
            reward += 100
        else:
            reward += human0_wait_penalty

        # human 1
        curr_human1_needed_obj = int(state[-1])

        deliver_success = (action == self.n_objs + 1) and \
                          (sum(next_objs_in_basket) < sum(objs_in_basket))

        # Deliver a good tool for human 1
        if objs_in_basket[curr_human1_needed_obj] > 0 and deliver_success:
            reward += 100
        else:
            reward += human1_wait_penalty

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
        # which object is waited for - human 0: [n_objs]
        # which object is waited for - human 1: [n_objs]

        done = False

        objs_in_basket = new_state[4:4+self.n_objs]
        objs_in_table = new_state[4+self.n_objs:4+2*self.n_objs]

        # the basket is empty and all objects are not on the table
        if sum(objs_in_basket) == 0 and sum(objs_in_table) == self.n_objs:
            done = True

        return done

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action[0]
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
    env = SpeedyToolDeliveryEnv(human_speeds=[10, 20])
    obs = env.reset()

    print(env.step(0)[1])
    print(env.step(0)[1])
    print(env.step(3)[1])
    print(env.step(4)[1])
    print(env.step(1)[1])
    print(env.step(1)[1])
    print(env.step(3)[1])
    print(env.step(4)[1])
    print(env.step(2)[1])
    print(env.step(2)[1])
    print(env.step(3)[1])
    print(env.step(4)[1])