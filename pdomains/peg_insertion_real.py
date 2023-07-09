import numpy as np
import gym
from gym import spaces
import copy
import rospy

from pdomains.robot_utils.robot_interface import RobotInterface

import yaml
with open('robot_utils/params.yaml', 'r') as yml:
    config = yaml.safe_load(yml)

neutral_pose     = config["neutral_pose"]
above_hole_pose  = config["above_hole_pose"]
hole_pose        = config["hole_pose"]


class PegInsertionEnv(gym.Env):
    """
    This class implements the cartesian velocity control environment.
    Same interface as an OpenAI gym environment.
    """

    def __init__(self, return_state=False):

        self.action_dim = 3
        high_action = np.ones(self.action_dim)  # delta_x, delta_y, delta_z
        self.action_space = spaces.Box(-high_action, high_action)

        self.observation_space = gym.spaces.Box(
            shape=(9, ), low=-np.inf, high=np.inf, dtype=np.float32
        )

        self.action_scaler = 0.02

        # Connect to the robot
        observation_options = ["wrist_ft", "cartesian"]
        self.ur5e = RobotInterface(observation_options=observation_options)

        self.speed_normal = 0.01
        self.speed_slow   = 0.005

        self.episode_cnt = 0

    def reset(self):
        """
        Go to a random position that touches the hole (to reduce the vibration)
        """
        self.ur5e.stop_robot()
        rospy.sleep(0.1)

        self.episode_cnt = 0

        print("Go to above hole pose")
        self.ur5e.go_to_cartesian_pose(neutral_pose, speed=self.speed_normal)

        print("Go to start pose")
        # TODO: randomize the start pose
        self.ur5e.go_to_cartesian_pose(above_hole_pose, speed=self.speed_slow)

        return self._process_obs()

    def _process_obs(self):
        """
        observation include: arm_tip_xyz, force_xyz, torque_xyz
        Need to convert all into the hole coordinate
        """
        observations = []

        arm_tip_xyz = self.ur5e.get_cartesian_state()
        observations.extend(arm_tip_xyz)

        f_x, f_y, f_z, t_x, t_y, t_z = self.ur5e.get_wrist_ft(averaged_number=0)

        observations.extend([f_x, f_y, f_z])
        observations.extend([t_x, t_y, t_z])

        # TODO: convert to hole coordinate

        return copy.deepcopy(observations)

    def step(self, action):
        """
        action (np.array): (delta_x, delta_y, delta_z)
        """

        # Clip the action
        action = np.clip(action, -1, 1) * self.action_scaler

        # Calculate the desired pose
        current_pose = self.ur5e.get_cartesian_state()
        desired_pose = current_pose + action

        # TODO: add desired orientation here
        breakpoint()

        # Send request
        self.ur5e.go_to_cartesian_pose(desired_pose, speed=self.speed_slow)

        rew = self._reward()
        done = self._done()

        self.episode_cnt += 1

        return self._process_obs(), rew, done, {}

    def _reward(self):
        return 0.0

    def _done(self):
        return False

    def close(self):
        pass

    def render(self, mode='human'):
        pass