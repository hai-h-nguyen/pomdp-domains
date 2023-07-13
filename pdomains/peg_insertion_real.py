import numpy as np
import gym
from gym import spaces
import copy
import rospy
import tf

from pdomains.robot_utils.robot_interface import RobotInterface
import pdomains.robot_utils.transform_utils as T

import yaml
with open('/root/o2ac-ur/catkin_ws/src/hai_ws/pomdp-domains/scripts/params.yaml', 'r') as yml:
    config = yaml.safe_load(yml)

HOLE_CENTER_POSE = config["hole_center_pose"]

HOLE_RADIUS = config["hole_radius"]
OUTER_RADIUS = config["outer_radius"]

X_THRES = config["x_threshold"]
Y_THRES = config["y_threshold"]
Z_THRES = config["z_threshold"]
RESET_XY_THRES = config["reset_xy_threshold"]

TIP2HOLE_OFFSET_Z = config["tip2hole_offset_z"]

MAX_FORCE = config["max_force"]
MAX_TORQUE = config["max_torque"]

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
        observation_options = ["cartesian", "wrist_ft_filtered"]
        self.ur5e = RobotInterface(observation_options=observation_options)

        self.ur5e.switch_controllers("moveit")

        self.speed_normal = 0.05
        self.speed_slow   = 0.005

        self.episode_cnt = 0

        # Forces and torques are defined at the tool0_control coordinate
        listener = tf.TransformListener()
        listener.waitForTransform('/hole_coordinate','/tool0_controller', rospy.Time(), rospy.Duration(4.0))
        (trans, rot) = listener.lookupTransform('/hole_coordinate', '/tool0_controller', rospy.Time(0))
        self.tool0_in_hole = T.pose2mat((trans, rot))

        # Positions are defined at the world coordinate
        listener.waitForTransform('/hole_coordinate','/world', rospy.Time(), rospy.Duration(4.0))
        (trans, rot) = listener.lookupTransform('/hole_coordinate', '/world', rospy.Time(0))
        self.world_in_hole = T.pose2mat((trans, rot))

    def reset(self):
        """
        Go to a random position that touches the hole (to reduce the vibration)
        """
        # self.ur5e.stop_robot()
        # rospy.sleep(0.1)

        self.episode_cnt = 0

        print("Go to right above hole center using the same height")
        current_height = self.ur5e.get_cartesian_state()[2]
        temp_center_hose = np.copy(HOLE_CENTER_POSE)
        temp_center_hose[2] = current_height
        self.ur5e.go_to_cartesian_pose(temp_center_hose, speed=self.speed_normal)

        print("Go to right above hole center using the hole height")
        self.ur5e.go_to_cartesian_pose(HOLE_CENTER_POSE, speed=self.speed_normal)        

        print("Go to random position")
        random_pose = self._randomize_starting_pos(HOLE_CENTER_POSE)
        self.ur5e.go_to_cartesian_pose(random_pose, speed=self.speed_slow)

        obs, _, _ = self._process_obs()
        return obs

    def _randomize_starting_pos(self, hole_pose):
        center_x, center_y = hole_pose[0], hole_pose[1]

        random_angle = 2*np.pi*np.random.rand()
        random_radius = HOLE_RADIUS + np.random.rand()*(OUTER_RADIUS - HOLE_RADIUS)

        random_x = center_x + random_radius * np.sin(random_angle)
        random_y = center_y + random_radius * np.cos(random_angle)

        ret = [random_x, random_y] + HOLE_CENTER_POSE[2:]

        return ret

    def _process_obs(self):
        """
        observation include: arm_tip_xyz, force_xyz, torque_xyz
        Need to convert all into the hole coordinate
        """
        observations = []
        terminate = False
        success = False

        arm_tip_in_world = self.ur5e.get_cartesian_state()
        arm_tip_xyz = arm_tip_in_world[:3]
        arm_tip_rot = arm_tip_in_world[3:]
        arm_tip_pose_in_world = T.pose2mat((arm_tip_xyz, arm_tip_rot))
        arm_tip_pose_in_hole = T.pose_in_A_to_pose_in_B(arm_tip_pose_in_world, self.world_in_hole)
        arm_tip_rel_pos, arm_tip_rel_quat = T.mat2pose(arm_tip_pose_in_hole)

        arm_tip_rel_pos[2] -= TIP2HOLE_OFFSET_Z

        norm_arm_tip_rel_pos = np.array(arm_tip_rel_pos) / RESET_XY_THRES
        observations.extend(list(norm_arm_tip_rel_pos))

        # check if success
        x_cond = abs(arm_tip_rel_pos[0]) <= X_THRES
        y_cond = abs(arm_tip_rel_pos[1]) <= Y_THRES
        z_cond = arm_tip_rel_pos[2] < -Z_THRES

        positional_success = x_cond and y_cond and z_cond

        # check terminate early if going too far
        x_cond = abs(arm_tip_rel_pos[0]) > RESET_XY_THRES
        y_cond = abs(arm_tip_rel_pos[1]) > RESET_XY_THRES

        terminate = x_cond or y_cond
        if terminate:
            print(f"Terminate due to moving away > x:{x_cond} y:{y_cond}")

        f_x, f_y, f_z, t_x, t_y, t_z = self.ur5e.get_wrist_ft_filtered()
        forces = np.array([f_x, f_y, f_z])
        torques = np.array([t_x, t_y, t_z])

        assert self.tool0_in_hole is not None
        new_forces, new_torques = T.force_in_A_to_force_in_B(forces, torques, self.tool0_in_hole)

        # check if force on Z is too much
        if not terminate:
            terminate = new_forces[2] > float(MAX_FORCE * 0.75)
            if terminate:
                print(f"Terminate due to force z > {float(MAX_FORCE * 0.75)}")
                print("Go up to relax")
                current_pose = self.ur5e.get_cartesian_state()
                current_pose[2] += 0.02
                self.ur5e.go_to_cartesian_pose(current_pose, speed=self.speed_normal)

        fx_cond = abs(new_forces[0]) < 1.5
        fy_cond = abs(new_forces[1]) < 1.5
        fz_cond = new_forces[2] > 5.0

        force_success = fx_cond and fy_cond and fz_cond
        success = positional_success and force_success

        # if not success:
            # print(new_forces, arm_tip_rel_pos)
        if success:
            print(f"Succeed!w {new_forces} {arm_tip_rel_pos}")

        norm_new_forces = np.array(new_forces) / MAX_FORCE
        norm_new_torques = np.array(new_torques) / MAX_TORQUE

        observations.extend(list(norm_new_forces))
        observations.extend(list(norm_new_torques))

        observations = np.array(observations)

        return copy.deepcopy(observations), terminate, success

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

        self.episode_cnt += 1

        obs, terminate, success = self._process_obs()

        done = terminate or success
        rew = float(success)

        return obs, rew, done, {}

    def close(self):
        pass

    def render(self, mode='human'):
        pass