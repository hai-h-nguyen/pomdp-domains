# -*- coding: utf-8 -*-

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper


class PegInsertionEnv(gym.Env):
    def __init__(self, rendering=False, seed=0, peg_type="square", return_state=False):

        # Get controller config
        controller_config = load_controller_config(default_controller="IK_POSE")

        self.action_scaler = 0.02
        if peg_type == "triangle":
            robot = "SoftUR5eTriangle"
        elif peg_type == "square":
            robot = "SoftUR5eSquare"
        elif peg_type == "pentagon":
            robot = "SoftUR5ePentagon"
        elif peg_type == "hexagon":
            robot = "SoftUR5eHexagon"
        elif peg_type == "round":
            robot = "SoftUR5eRound"
        else:
            raise ValueError("Invalid peg type: {}".format(peg_type))

        # Create argument configuration
        config = {
            "env_name": "SoftPegInHole",
            "robots": robot,
            "controller_configs": controller_config,
        }

        self.rendering = rendering
        self.return_state = return_state

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

        self.action_dim = 2
        high_action = np.ones(self.action_dim)  # delta_x, delta_z
        self.action_space = spaces.Box(-high_action, high_action)

        self.obs_dims = [0, 2, 3, 5]
        # relative x, relative z, f_x, f_z
        self.observation_space = gym.spaces.Box(
            shape=(len(self.obs_dims),), low=-np.inf, high=np.inf, dtype=np.float32
        )

        self.state_data = None

        self.seed(seed=seed)

    def get_state(self):
        return self.state_data

    def seed(self, seed=0):
        self.np_random, seed_ = seeding.np_random(seed)
        return seed_

    def _process_obs(self, obs):
        """
        select features to create the observation
        """
        assert len(obs["all_sensors"]) == 25
        all_data = obs["all_sensors"]

        peg_2_hole_xz = all_data[:3][[0, 2]]

        force_xz = all_data[-9:][[0, 2]]

        self.state_data = np.concatenate((peg_2_hole_xz, force_xz))

        if self.return_state:
            return self.state_data.copy()
        else:
            return all_data[-9:][self.obs_dims].copy()

    def step(self, action):
        action = self._process_action(action)
        obs, reward, done, info = self.core_env.step(action)

        info = {}

        info["success"] = reward > 0.0

        if reward > 0.0 or obs["all_sensors"][-10]:
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

        self.state_data = None

        if self.rendering:
            self.core_env.render()

        action = self.action_space.sample()

        # not pushing down on the Z axis
        if action[self.action_dim - 1] < 0.0:
            action[self.action_dim - 1] = 0.0

        action = self._process_action(action)

        for _ in range(5):
            obs, _, _, _ = self.core_env.step(action)

        if self.rendering:
            self.core_env.render()

        return self._process_obs(obs)

    def _process_action(self, action):
        """
        zero out the gripper action and the rotations along XY axes
        """
        sent_action = np.zeros(6)
        sent_action[0] = action[0]  # delta x
        sent_action[2] = action[1]  # delta z

        return sent_action*self.action_scaler

    def close(self):
        pass
