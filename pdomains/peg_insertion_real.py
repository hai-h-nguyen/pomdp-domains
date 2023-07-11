import numpy as np
import gym
from gym import spaces
import copy
import rospy

from pdomains.robot_utils.robot_interface import RobotInterface

import yaml
with open('pdomains/robot_utils/params.yaml', 'r') as yml:
    config = yaml.safe_load(yml)

hole_center_pose = config["hole_center_pose"]

hole_radius = config["hole_radius"]
outer_radius = config["outer_radius"]

x_threshold = config["x_threshold"]
y_threshold = config["y_threshold"]
z_threshold = config["z_threshold"]

force_offsets = config["force_offsets"]
torque_offsets = config["torque_offsets"]

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
        observation_options = ["cartesian", "wrist_ft"]
        self.ur5e = RobotInterface(observation_options=observation_options)

        self.ur5e.switch_controllers("moveit")

        self.speed_normal = 0.025
        self.speed_slow   = 0.005

        self.episode_cnt = 0

    def reset(self):
        """
        Go to a random position that touches the hole (to reduce the vibration)
        """
        # self.ur5e.stop_robot()
        # rospy.sleep(0.1)

        self.episode_cnt = 0

        # print("Go to right above hole center")
        # self.ur5e.go_to_cartesian_pose(hole_center_pose, speed=self.speed_normal)

        # print("Go to random position")
        # random_pose = self._randomize_starting_pos(hole_center_pose)
        # self.ur5e.go_to_cartesian_pose(random_pose, speed=self.speed_slow)

        # print("Go down to touch the hole plane")

        # print("Go to a random position")
        # TODO: randomize the start pose
        # random_pose = hole_center_pose
        # self.ur5e.go_to_cartesian_pose(random_pose, speed=self.speed_slow)

        return self._process_obs()

    def _randomize_starting_pos(self, hole_pose):
        center_x, center_y = hole_pose[0], hole_pose[1]

        random_angle = 2*np.pi*np.random.rand()
        random_radius = hole_radius + np.random.rand()*(outer_radius - hole_radius)

        random_x = center_x + random_radius * np.sin(random_angle)
        random_y = center_y + random_radius * np.cos(random_angle)

        ret = [random_x, random_y] + hole_center_pose[2:]

        return ret

    def _process_obs(self):
        """
        observation include: arm_tip_xyz, force_xyz, torque_xyz
        Need to convert all into the hole coordinate
        """
        observations = []

        arm_tip_xyz = self.ur5e.get_cartesian_state()

        observations.extend(arm_tip_xyz)

        f_x, f_y, f_z, t_x, t_y, t_z = self.ur5e.get_wrist_ft(averaged_number=15)

        print(f_x, f_y, f_z)
        print(t_x, t_y, t_z)
        print("-------")

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
        pad_action = np.zeros(7)  # delta_xyz, quat_orientation

        pad_action[:3] = action

        # Calculate the desired pose
        current_pose = self.ur5e.get_cartesian_state()
        desired_pose = current_pose + pad_action  # no changing the current orientation

        # Send request
        self.ur5e.go_to_cartesian_pose(desired_pose, speed=self.speed_normal)

        rew = self._reward()
        done = self._done()

        if rew > 0:
            done = True

        self.episode_cnt += 1

        return self._process_obs(), rew, done, {}

    def _reward(self):
        current_pos = self.ur5e.get_cartesian_state()
        current_pos_x = current_pos[0]
        current_pos_y = current_pos[1]
        current_pos_z = current_pos[2]

        # print(abs(current_pos_x - hole_center_pose[0]))
        # print(abs(current_pos_y - hole_center_pose[1]))
        # print(current_pos_z - hole_center_pose[2])
        # print("--------------------")

        x_cond = abs(current_pos_x - hole_center_pose[0]) <= x_threshold
        y_cond = abs(current_pos_y - hole_center_pose[1]) <= y_threshold
        z_cond = current_pos_z - hole_center_pose[2] < -z_threshold

        return x_cond and y_cond and z_cond

    def _done(self):
        return False

    def close(self):
        pass

    def render(self, mode='human'):
        pass