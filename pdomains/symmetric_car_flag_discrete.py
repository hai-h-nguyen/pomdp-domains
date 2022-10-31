# -*- coding: utf-8 -*-

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering as visualize
import random

STEP_PENALTY = -0.01
FOUND_HEAVEN_REWARD = 1.0
FOUND_HELL_REWARD = -1.0

class CarEnv(gym.Env):
    def __init__(self, seed=0, rendering=False):
        self.max_position = 1.2
        self.min_position = -self.max_position

        self.setup_view = False

        self.min_action = -1.0
        self.max_action = 1.0

        self.heaven_position = 1.0
        self.hell_position = -1.0
        self.priest_position = 0.0

        self.delta = 0.2

        self.low_state = np.array([self.min_position, -1.0])
        self.high_state = np.array([self.max_position, 1.0])

        self.viewer = None
        self.show = rendering

        self.screen_width = 600
        self.screen_height = 400

        self.low_state = np.array(
            [self.min_position, -1.0], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, 1.0], dtype=np.float32
        )

        world_width = self.max_position - self.min_position
        self.scale = self.screen_width/world_width

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

        self.seed()

        self.max_ep_length = 50
        self.steps_taken = 0
        self.reached_heaven = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if np.isscalar(action):
            action = [action]

        position = self.state[0]

        self.steps_taken += 1

        position += action[0]*self.delta
        if (position >= self.max_position): position = self.max_position
        if (position <= self.min_position): position = self.min_position

        max_position = max(self.heaven_position, self.hell_position)
        min_position = min(self.heaven_position, self.hell_position)

        env_reward = STEP_PENALTY

        if (self.heaven_position > self.hell_position):
            if (position >= self.heaven_position):
                env_reward += FOUND_HEAVEN_REWARD
                self.reached_heaven = True

            if (position <= self.hell_position):
                env_reward += FOUND_HELL_REWARD

        if (self.heaven_position < self.hell_position):
            if (position <= self.heaven_position):
                env_reward += FOUND_HEAVEN_REWARD
                self.reached_heaven = True

            if (position >= self.hell_position):
                env_reward += FOUND_HELL_REWARD

        direction = 0.0
        if position == self.priest_position:
            if (self.heaven_position > self.hell_position):
                # Heaven on the right
                direction = 1.0
            else:
                # Heaven on the left
                direction = -1.0

        self.state = np.array([position, direction])

        if self.show:
            self.render()

        info = {}
        info["success"] = self.reached_heaven

        done = bool(
            position >= max_position or position <= min_position
        )

        return self.state, env_reward, done, info

    def render(self, mode='human'):
        self._setup_view()

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos-self.min_position) * self.scale, self._height(pos) * self.scale
        )

        return self.viewer.render(return_rgb_array=mode=='rgb_array')

    def reset(self):

        self.steps_taken = 0
        self.reached_heaven = False

        # Randomize the heaven/hell location
        if (self.np_random.randint(2) == 0):
            self.heaven_position = 1.0
        else:
            self.heaven_position = -1.0

        self.hell_position = -self.heaven_position

        if self.viewer is not None:
            self._draw_flags()

        range = [1, 2, 3, -1, -2, -3]
        pos = random.choice(range)
        self.state = np.array([pos*self.delta, 0.0])
        return np.array(self.state)

    def _height(self, xs):
        return .55 * np.ones_like(xs)

    def _draw_flags(self):
        scale = self.scale
        # Flag Heaven
        flagx = (abs(self.heaven_position)-self.min_position)*scale
        flagy1 = self._height(self.heaven_position)*scale
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)
        flag = visualize.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )

        # RED for hell
        if self.heaven_position > self.hell_position:
            flag.set_color(0.0, 1.0, 0)
        else:
            flag.set_color(1.0, 0.0, 0)

        self.viewer.add_geom(flag)

        # Flag Hell
        flagx = (-abs(self.heaven_position)-self.min_position)*scale
        flagy1 = self._height(self.hell_position)*scale
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)
        flag = visualize.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )

        # GREEN for heaven
        if self.heaven_position > self.hell_position:
            flag.set_color(1.0, 0.0, 0)
        else:
            flag.set_color(0.0, 1.0, 0)

        self.viewer.add_geom(flag)

        # BLUE for priest
        flagx = (self.priest_position-self.min_position)*scale
        flagy1 = self._height(self.priest_position)*scale
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)
        flag = visualize.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )
        flag.set_color(0.0, 0.0, 1.0)
        self.viewer.add_geom(flag)

    def _setup_view(self):
        if  not self.setup_view:
            self.viewer = visualize.Viewer(self.screen_width, self.screen_height)
            scale = self.scale
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = visualize.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10
            carwidth = 40
            carheight = 20

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = visualize.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(visualize.Transform(translation=(0, clearance)))
            self.cartrans = visualize.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = visualize.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                visualize.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = visualize.make_circle(carheight / 2.5)
            backwheel.add_attr(
                visualize.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)

            self._draw_flags()

            self.setup_view = True        

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
