# Based on https://github.com/allenai/advisor/blob/main/poisoneddoors_plugin/poisoneddoors_tasks.py

import random
from enum import Enum
from typing import Any, Tuple, Union, List, Optional, Dict

import gym
import numpy as np
from gym.utils import seeding
from gym.spaces import Box, Discrete


def get_combination(nactions: int, combination_length: int):
    s = random.getstate()
    random.seed(combination_length)
    comb = [random.randint(0, nactions - 1) for _ in range(combination_length)]
    random.setstate(s)
    return comb


class PoisonedEnvStates(Enum):
    choosing_door = 0
    entering_pass_start = 1
    entering_pass_cont = 2
    done = 3


class PoisonedDoorsEnv(gym.Env):
    def __init__(self, num_doors=4, combination_length=10, seed=0):
        self.num_doors = num_doors
        self.combination_length = combination_length

        self.combination = get_combination(
            nactions=3, combination_length=self.combination_length
        )

        self.combination_index = 0
        self.max_comb_index = 0

        self.current_state = PoisonedEnvStates.choosing_door
        self.chosen_door: Optional[int] = None
        self.good_door_ind: Optional[int] = None

        self.nstates = len(PoisonedEnvStates)

        self.action_space = Discrete(len(self.action_names()))
        self.observation_space = Box(low=0, high=self.nstates - 1, shape=(1,), dtype=int,)

        self.seed(seed)

    @classmethod
    def class_action_names(cls, num_doors: int):
        return ("c0", "c1", "c2") + tuple(str(i) for i in range(num_doors))

    def action_names(self):
        return self.class_action_names(num_doors=self.num_doors)

    def internal_reset(self, door_ind: int):
        assert 1 <= door_ind < self.num_doors
        self.good_door_ind = door_ind
        # print(self.good_door_ind)
        self.chosen_door = None
        self.current_state = PoisonedEnvStates.choosing_door
        self.combination_index = 0
        self.max_comb_index = 0

    def reset(self):
        seed = self.np_random.randint(0, 2 ** 31 - 1)
        self.internal_reset(door_ind=1 + (seed % (self.num_doors - 1)))
        return self.get_observation()

    def get_observation(self):
        return np.array([int(self.current_state.value)])

    def is_done(self):
        return self.current_state == PoisonedEnvStates.done

    def step(self, action: int) -> float:
        if action < 3 or self.current_state != self.current_state.choosing_door:
            if self.chosen_door is None:
                reward = 0.0
            else:
                assert self.chosen_door == 0, "Stepping when done."

                correct = self.combination[self.combination_index] == action

                if correct:
                    self.combination_index += 1
                    self.max_comb_index = max(
                        self.combination_index, self.max_comb_index
                    )
                else:
                    self.combination_index = 0

                if correct:
                    self.current_state = PoisonedEnvStates.entering_pass_cont
                elif not correct:
                    self.current_state = PoisonedEnvStates.done

                if self.combination_index >= len(self.combination):
                    self.current_state = PoisonedEnvStates.done
                    reward = 1.0
                reward = 0.0
        elif action == 3:
            self.chosen_door = 0
            self.combination_index = 0
            self.current_state = PoisonedEnvStates.entering_pass_start
            reward = 0.0
        else:
            self.current_state = PoisonedEnvStates.done
            self.chosen_door = action - 3
            reward = 2.0 * (1 if self.good_door_ind == action - 3 else -1)

        done = self.is_done()

        return self.get_observation(), reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
