import numpy as np
import gym
from gym import spaces
import copy
import rospy
import tf

from pdomains.robot_utils.robot_interface import RobotInterface
import pdomains.robot_utils.transform_utils as T

from std_srvs.srv import Empty

import yaml
with open('/root/o2ac-ur/catkin_ws/src/hai_ws/pomdp-domains/scripts/params_square.yaml', 'r') as yml:
    config = yaml.safe_load(yml)

HOLE_CENTER_POSE = config["hole_center_pose"]

HOLE_RADIUS = config["hole_radius"]
OUTER_RADIUS = config["outer_radius"]

X_THRES = config["x_threshold"]
Y_THRES = config["y_threshold"]
Z_THRES = config["z_threshold"]
RESET_XY_THRES = config["reset_xy_threshold"]
RESET_Z_THRES = config["reset_z_threshold"]

TIP2HOLE_OFFSET_Z = config["tip2hole_offset_z"]

FORCE_NORMALIZER = config["force_normalizer"]
TORQUE_NORMALIZER = config["torque_normalizer"]
RESET_FORCE_Z_THRES = config["reset_fz_threshold"]


class PegInsertionEnv(gym.Env):
    def __init__(self):

        self.action_dim = 3
        high_action = np.ones(self.action_dim)  # delta_x, delta_y, delta_z
        self.action_space = spaces.Box(-high_action, high_action)

        self.observation_space = gym.spaces.Box(
            shape=(9, ), low=-np.inf, high=np.inf, dtype=np.float32
        )

        self.action_scaler = np.array([0.01, 0.01, 0.0025])

        # Connect to the robot
        observation_options = ["wrist_ft_filtered", "detector"]
        self.ur5e = RobotInterface(observation_options=observation_options)

        self.ur5e.switch_controllers("moveit")

        self.speed_normal = 0.01
        self.speed_slow   = 0.006

        self.step_cnt = 0

        self.zero_ft_serv = rospy.ServiceProxy("/wrench/filtered/zero_ftsensor", Empty)

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

        self.print_ft_once = False
        
        self.success_episode = 0
        self.episode_cnt = 0

    def reset(self, eval=False):
        """
        Go to a random position that touches the hole (to reduce the vibration)
        """
        self.step_cnt = 0

        if eval:
            print(f"Success/Total: {self.success_episode}/{self.episode_cnt}")
            self.episode_cnt += 1

        if self._is_force_z_large():
            print("Go up to relax")
            current_pose = self.ur5e.get_cartesian_state()
            current_pose[2] += 0.01
            self.ur5e.go_to_cartesian_pose(current_pose, speed=self.speed_normal)
        
        # Use these two lines to get the hole pose
        # print(self.ur5e.get_cartesian_state())
        # breakpoint()
        print("Go to right above hole center using the current height")
        current_height = self.ur5e.get_cartesian_state()[2]
        temp_center_hose = np.copy(HOLE_CENTER_POSE)
        temp_center_hose[2] = current_height
        self.ur5e.go_to_cartesian_pose(temp_center_hose, speed=self.speed_normal)

        print("Go to hole center using the hole height")
        self.ur5e.go_to_cartesian_pose(HOLE_CENTER_POSE, speed=self.speed_normal)

        print("Call a service to update the F/T offsets")
        rospy.wait_for_service("/wrench/filtered/zero_ftsensor")
        self.zero_ft_serv()
        self.print_ft_once = False

        print("Go to random position")
        random_pose = self._randomize_starting_pos(HOLE_CENTER_POSE, eval)
        self.ur5e.go_to_cartesian_pose(random_pose, speed=self.speed_slow)

        obs, _, _, _ = self._process_obs()
        return obs

    def _randomize_starting_pos(self, hole_pose, eval=False):
        random_angle = 2*np.pi*np.random.rand()
        if eval:
            hole_radius = 0.015
        else:
            hole_radius = HOLE_RADIUS
        random_radius = hole_radius + np.random.rand()*(OUTER_RADIUS - hole_radius)

        random_x = hole_pose[0] + random_radius * np.sin(random_angle)
        random_y = hole_pose[1] + random_radius * np.cos(random_angle)

        return [random_x, random_y] + HOLE_CENTER_POSE[2:]

    def _process_obs(self, action=0.0):
        """
        observation include: arm_tip_xyz, force_xyz, torque_xyz
        in the hole coordinate
        """
        observations = []
        terminate = False
        success = False
        penalty = 0.0

        arm_tip_in_world = self.ur5e.get_cartesian_state()
        arm_tip_xyz = arm_tip_in_world[:3]
        arm_tip_rot = arm_tip_in_world[3:]
        arm_tip_pose_in_world = T.pose2mat((arm_tip_xyz, arm_tip_rot))
        arm_tip_pose_in_hole = T.pose_in_A_to_pose_in_B(arm_tip_pose_in_world, self.world_in_hole)
        arm_tip_pos_in_hole, arm_tip_quat_in_hole = T.mat2pose(arm_tip_pose_in_hole)

        # Use these two lines to get the hole pose
        # print(arm_tip_pos_in_hole)
        # breakpoint()

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
        z_cond = abs(arm_tip_pos_in_hole[2]) > RESET_Z_THRES

        terminate = x_cond or y_cond or z_cond
        if terminate:
            print(f"Terminate due to moving away x:{x_cond} y:{y_cond} z:{z_cond}")
            penalty += -5.0

        f_x, f_y, f_z, t_x, t_y, t_z = self.ur5e.get_wrist_ft_filtered()
        forces_in_tool0 = np.array([f_x, f_y, f_z])
        torques_in_tool0 = np.array([t_x, t_y, t_z])

        forces_in_hole, torques_in_hole = T.force_in_A_to_force_in_B(forces_in_tool0, torques_in_tool0, self.tool0_in_hole)

        # print("F/T", forces_in_hole, torques_in_hole)

        if not self.print_ft_once:
            print("F/T", forces_in_hole, torques_in_hole)
            self.print_ft_once = True

        # check if force on Z is too much
        if not terminate:
            terminate = abs(forces_in_hole[2]) > RESET_FORCE_Z_THRES
            if terminate:
                penalty += -5.0
        
        # xy_dist_2_hole = np.sqrt(arm_tip_pos_in_hole[0]**2 + arm_tip_pos_in_hole[1]**2)
        # if xy_dist_2_hole <= 0.005:
        #     torque_penalty = -np.linalg.norm(np.array(forces_in_hole[:2]))
        #     print("Torque penalty proximity", torque_penalty)
        #     force_penalty += torque_penalty

        if forces_in_hole[2] < 0.0:
            penalty += -0.1
        
        # fx_cond = abs(forces_in_hole[0]) < 1.5
        # fy_cond = abs(forces_in_hole[1]) < 1.5
        # fz_cond = forces_in_hole[2] > 5.0

        # force_success = fx_cond and fy_cond and fz_cond

        # tx_cond = abs(torques_in_hole[0]) < 0.5
        # ty_cond = abs(torques_in_hole[1]) < 0.5
        # tz_cond = abs(torques_in_hole[2]) < 0.5

        # torque_success = tx_cond and ty_cond and tz_cond
        success = positional_success

        force_mag = np.linalg.norm(forces_in_hole)
        theta = np.arccos(forces_in_hole[2]/force_mag)

        if success:
            visual_success = not self.ur5e.is_detect_red()
            success = success and visual_success and (theta*57.3 < 30.0)

        if success:
            print(f"Succeed! w/ {forces_in_hole} {arm_tip_pos_in_hole} {torques_in_hole} {theta*57.3}")
        else:
            print(self.step_cnt, action, arm_tip_pos_in_hole,
                  forces_in_hole, torques_in_hole, round(theta*57.3, 1))

        norm_forces_in_hole = np.array(forces_in_hole) / FORCE_NORMALIZER
        norm_torques_in_hole = np.array(torques_in_hole) / TORQUE_NORMALIZER

        observations.extend(list(norm_forces_in_hole))
        observations.extend(list(norm_torques_in_hole))

        observations = np.array(observations).astype(np.float32)

        return copy.deepcopy(observations), terminate, success, penalty

    def step(self, action):
        """
        action (np.array): (delta_x, delta_y, delta_z)
        """

        # Clip the action
        raw_action = action.copy()
        action = np.clip(action, -1.0, 1.0) * self.action_scaler
        pad_action = np.zeros(7)  # delta_xyz, quat_orientation
        
        pad_action[:3] = action

        # Calculate the desired pose
        current_pose = self.ur5e.get_cartesian_state()
        desired_pose = current_pose + pad_action  # keep the orientation

        # Send request
        self.ur5e.go_to_cartesian_pose(desired_pose, speed=self.speed_normal)

        self.step_cnt += 1

        obs, early_terminate, success, penalty = self._process_obs(raw_action)

        done = early_terminate or success
        rew = float(success) + penalty

        info = {}

        info["success"] = success

        if success:
            self.success_episode += 1

        return obs, rew, done, info

    def close(self):
        pass

    def render(self, mode='human'):
        pass

    def _is_force_z_large(self):
        """
        If the force_z is current large than a threshold and the agent is
        trying to go down more, ignore the request on the height
        the environment will be reset later
        """
        f_x, f_y, f_z, t_x, t_y, t_z = self.ur5e.get_wrist_ft_filtered()
        forces_in_tool0 = np.array([f_x, f_y, f_z])
        torques_in_tool0 = np.array([t_x, t_y, t_z])

        forces_in_hole, torques_in_hole = T.force_in_A_to_force_in_B(forces_in_tool0, torques_in_tool0, self.tool0_in_hole)

        # check if force on Z is too much
        force_z_large = abs(forces_in_hole[2]) > RESET_FORCE_Z_THRES

        if force_z_large:
            print(f"Force Z too large {forces_in_hole[2]} > {RESET_FORCE_Z_THRES}")

        return force_z_large
