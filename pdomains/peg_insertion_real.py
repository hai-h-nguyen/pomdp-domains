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
RESET_FORCE_Z_THRES = config["reset_fz_threshold"]


class PegInsertionEnv(gym.Env):
    def __init__(self):

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

        self.step_cnt = 0

        # Forces and torques are defined at the tool0_control coordinate
        target_coordinate = '/hole_coordinate'
        listener = tf.TransformListener()
        listener.waitForTransform(target_coordinate,'/tool0_controller', rospy.Time(), rospy.Duration(4.0))
        (trans, rot) = listener.lookupTransform(target_coordinate, '/tool0_controller', rospy.Time(0))
        self.tool0_in_hole = T.pose2mat((trans, rot))

        # Positions are defined at the world coordinate
        listener.waitForTransform(target_coordinate,'/world', rospy.Time(), rospy.Duration(4.0))
        (trans, rot) = listener.lookupTransform(target_coordinate, '/world', rospy.Time(0))
        self.world_in_hole = T.pose2mat((trans, rot))

    def reset(self):
        """
        Go to a random position that touches the hole (to reduce the vibration)
        """
        self.step_cnt = 0

        print("Go to right above hole center using the current height")
        current_height = self.ur5e.get_cartesian_state()[2]
        temp_center_hose = np.copy(HOLE_CENTER_POSE)
        temp_center_hose[2] = current_height
        self.ur5e.go_to_cartesian_pose(temp_center_hose, speed=self.speed_normal)

        print("Go to hole center using the hole height")
        self.ur5e.go_to_cartesian_pose(HOLE_CENTER_POSE, speed=self.speed_normal)        

        print("Go to random position")
        random_pose = self._randomize_starting_pos(HOLE_CENTER_POSE)
        self.ur5e.go_to_cartesian_pose(random_pose, speed=self.speed_slow)

        obs, _, _ = self._process_obs()
        return obs

    def _randomize_starting_pos(self, hole_pose):
        random_angle = 2*np.pi*np.random.rand()
        random_radius = HOLE_RADIUS + np.random.rand()*(OUTER_RADIUS - HOLE_RADIUS)

        random_x = hole_pose[0] + random_radius * np.sin(random_angle)
        random_y = hole_pose[1] + random_radius * np.cos(random_angle)

        return [random_x, random_y] + HOLE_CENTER_POSE[2:]

    def _process_obs(self):
        """
        observation include: arm_tip_xyz, force_xyz, torque_xyz
        in the hole coordinate
        """
        observations = []
        terminate = False
        success = False

        arm_tip_in_world = self.ur5e.get_cartesian_state()
        arm_tip_xyz = arm_tip_in_world[:3]
        arm_tip_rot = arm_tip_in_world[3:]
        arm_tip_pose_in_world = T.pose2mat((arm_tip_xyz, arm_tip_rot))
        arm_tip_pose_in_hole = T.pose_in_A_to_pose_in_B(arm_tip_pose_in_world, self.world_in_hole)
        arm_tip_pos_in_hole, arm_tip_quat_in_hole = T.mat2pose(arm_tip_pose_in_hole)

        arm_tip_pos_in_hole[2] -= TIP2HOLE_OFFSET_Z

        # print(arm_tip_pos_in_hole)

        norm_arm_tip_pose_in_hole = np.array(arm_tip_pos_in_hole) / RESET_XY_THRES
        observations.extend(list(norm_arm_tip_pose_in_hole))

        # check if success
        x_cond = abs(arm_tip_pos_in_hole[0]) <= X_THRES
        y_cond = abs(arm_tip_pos_in_hole[1]) <= Y_THRES
        z_cond = arm_tip_pos_in_hole[2] < -Z_THRES

        positional_success = x_cond and y_cond and z_cond

        # check terminate early if going too far
        x_cond = abs(arm_tip_pos_in_hole[0]) > RESET_XY_THRES
        y_cond = abs(arm_tip_pos_in_hole[1]) > RESET_XY_THRES

        terminate = x_cond or y_cond
        if terminate:
            print(f"Terminate due to moving away > x:{x_cond} y:{y_cond}")

        f_x, f_y, f_z, t_x, t_y, t_z = self.ur5e.get_wrist_ft_filtered()
        forces_in_tool0 = np.array([f_x, f_y, f_z])
        torques_in_tool0 = np.array([t_x, t_y, t_z])

        forces_in_hole, torques_in_hole = T.force_in_A_to_force_in_B(forces_in_tool0, torques_in_tool0, self.tool0_in_hole)

        # print(forces_in_hole)

        # check if force on Z is too much
        if not terminate:
            terminate = forces_in_hole[2] > RESET_FORCE_Z_THRES
            if terminate:
                print(f"Terminate due to force {forces_in_hole[2]} > {RESET_FORCE_Z_THRES}")
                print("Go up to relax")
                current_pose = self.ur5e.get_cartesian_state()
                current_pose[2] += 0.02
                self.ur5e.go_to_cartesian_pose(current_pose, speed=self.speed_normal)

        fx_cond = abs(forces_in_hole[0]) < 1.5
        fy_cond = abs(forces_in_hole[1]) < 1.5
        fz_cond = forces_in_hole[2] > 5.0

        force_success = fx_cond and fy_cond and fz_cond
        success = positional_success and force_success

        # if not success:
            # print(new_forces, arm_tip_rel_pos)
        if success:
            print(f"Succeed!w {forces_in_hole} {arm_tip_pos_in_hole}")

        norm_forces_in_hole = np.array(forces_in_hole) / MAX_FORCE
        norm_torques_in_hole = np.array(torques_in_hole) / MAX_TORQUE

        observations.extend(list(norm_forces_in_hole))
        observations.extend(list(norm_torques_in_hole))

        observations = np.array(observations)

        return copy.deepcopy(observations), terminate, success

    def step(self, action):
        """
        action (np.array): (delta_x, delta_y, delta_z)
        """

        # Clip the action
        action = np.clip(action, -1.0, 1.0) * self.action_scaler
        pad_action = np.zeros(7)  # delta_xyz, quat_orientation

        pad_action[:3] = action

        # Calculate the desired pose
        current_pose = self.ur5e.get_cartesian_state()
        desired_pose = current_pose + pad_action  # keep the orientation

        # Send request
        self.ur5e.go_to_cartesian_pose(desired_pose, speed=self.speed_normal)

        self.step_cnt += 1

        obs, terminate, success = self._process_obs()

        done = terminate or success
        rew = float(success)

        info = {}

        info["success"] = success

        return obs, rew, done, {}

    def close(self):
        pass

    def render(self, mode='human'):
        pass