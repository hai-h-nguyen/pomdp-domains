# -*- coding: utf-8 -*-

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gym_minigrid import minigrid

import curses
import time
from typing import Optional, Tuple, Any, Union, cast


class LightHouse1DEnv(gym.Env):
    STEP_PENALTY = -0.01
    FOUND_TARGET_REWARD = 1.0
    DISCOUNT_FACTOR = 0.99

    GO_LEFT = 0
    GO_RIGHT = 1

    def __init__(self, world_length: int = 31, view_radius: int = 1, max_step: int = 100, **kwargs):

        assert((world_length - 1) % 2 == 0)
        assert(view_radius >= 1)

        self.world_length = world_length

        self.half_world_length = (world_length - 1) // 2
        self.view_radius = view_radius

        self.curses_screen: Optional[Any] = None

        self.current_position = self.half_world_length
        self.goal_position: Optional[int] = None
        self.wall_position: Optional[int] = None

        self.max_step = max_step
        self.num_steps_taken = 0
        self._found_target = False

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-np.ones(self.world_length + 2*self.view_radius + 1),
            high=np.ones(self.world_length + 2*self.view_radius + 1),
            dtype=np.float32
        )

        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32
        )

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

    def query_expert(self) -> int:
        if self.goal_position == 0:
            return [self.GO_LEFT]
        else:
            return [self.GO_RIGHT]

    def query_state(self) -> np.array:
        return np.array([self.current_position, self.goal_position])

    def reset(self, goal_position: Optional[bool] = None):
        self.num_steps_taken = 0
        self._found_target = False

        # 0: Left, world_dim-1: Right
        if goal_position is None:
            self.goal_position = self.np_seeded_random_gen.choice((0, self.world_length - 1))
        else:
            self.goal_position = goal_position
        self.wall_position = self.world_length - self.goal_position - 1

        if self.curses_screen is not None:
            curses.nocbreak()
            self.curses_screen.keypad(False)
            curses.echo()
            curses.endwin()

        self.curses_screen = None

        self.current_position = self.half_world_length

        return self.get_observation()

    def step(self, action: int) -> Tuple:
        assert 0 <= action < 2

        already_done = (self.num_steps_taken > self.max_step or self._found_target)

        assert (already_done is False), "Continue when already terminated"

        self.num_steps_taken += 1

        reward = self.STEP_PENALTY
        done = False

        delta = -1 if action == self.GO_LEFT else 1
        old = self.current_position
        new = min(max(delta + old, 0), self.world_length - 1)
        if new != old:
            self.current_position = new

        if self.current_position == self.goal_position:
            self._found_target = True
            reward += self.FOUND_TARGET_REWARD
            done = True
        elif self.num_steps_taken == self.max_step:
            reward = self.STEP_PENALTY / (1 - self.DISCOUNT_FACTOR)
            done = True

        return self.get_observation(), reward, done, {}

    def get_observation(self) -> np.array:
        # agent position
        encoded_pos = np.zeros(self.world_length)
        encoded_pos[self.current_position] = 1

        # agent two-sideed view
        twosided_view = np.zeros(2*self.view_radius + 1)
        abs_distance_to_goal = abs(self.current_position - self.goal_position)

        if abs_distance_to_goal <= self.view_radius:
            if self.goal_position == 0:
                twosided_view[self.view_radius - abs_distance_to_goal] = 1
            else:
                twosided_view[self.view_radius + abs_distance_to_goal] = 1

        abs_distance_to_wall = abs(self.current_position - self.wall_position)
        if abs_distance_to_wall <= self.view_radius:
            if self.goal_position == 0:
                twosided_view[self.view_radius + abs_distance_to_wall] = -1
            else:
                twosided_view[self.view_radius - abs_distance_to_wall] = -1

        return np.concatenate((encoded_pos, twosided_view))

    def render(self, mode="human", **kwargs):
        if mode == "human":
            space_list = ["_"] * self.world_length

            goal_ind = self.goal_position
            space_list[goal_ind] = "G"
            space_list[self.world_length - goal_ind - 1] = "W"
            space_list[self.current_position] = "X"

            if self.current_position + self.view_radius < self.world_length:
                space_list[self.current_position + self.view_radius] = "|"

            if self.current_position - self.view_radius > 0:
                space_list[self.current_position - self.view_radius] = "|"

            to_print = " ".join(space_list)

            if self.curses_screen is None:
                self.curses_screen = curses.initscr()

            self.curses_screen.addstr(0, 0, to_print)
            self.curses_screen.refresh()
        else:
            raise NotImplementedError("Unknown render mode {}.".format(mode))

        time.sleep(0.1)

    def close(self):
        if self.curses_screen is not None:
            curses.nocbreak()
            self.curses_screen.keypad(False)
            curses.echo()
            curses.endwin()
