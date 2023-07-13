#!/usr/bin/env python3

import copy
import numpy as np, cv2

import rospy, tf
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayDimension
from std_srvs.srv import Empty
from o2ac_msgs.srv import *
from o2ac_msgs.msg import *
from geometry_msgs.msg import Pose
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from tf.transformations import euler_from_quaternion
from collections import deque

import time # debugging
import actionlib
import actionlib_tutorials.msg
import robotiq_msgs.msg
import moveit_msgs.msg
import moveit_commander
from controller_manager_msgs.srv import SwitchController
from o2ac_routines.robotiq_gripper import RobotiqGripper


def xyz_to_array(msg):
    return [msg.x, msg.y, msg.z]


class RobotInterface(object):
    """
    This class interfaces with the robot directly, and handles all communication
    """

    def __init__(self, observation_options=["cartesian"]):
        rospy.init_node('cartesian_velocity_env', disable_signals=True, log_level=rospy.DEBUG)

        self.observation_options = observation_options
        # Attributes
        self._num_joints = 6
        self._zero_joints = [0.0 for j in range(self._num_joints)]

        # Offset for gripper (IK)
        self._end_effector_link = "robotiq_85_flexwrist_tip_link_with_peg"
        self._end_effector_offset = [0.215, 0, 0]

        # Movement speed for planning
        self._moveit_vel_scale       = 0.01
        self._cartesian_speed_normal = 0.01
        self._cartesian_speed_slow   = 0.005

        # Publisher to send velocity command to joint_group_vel_controller
        self.joint_pub = rospy.Publisher('/joint_group_vel_controller/command',
                                        Float64MultiArray, queue_size=1)

        # Killing the node would mean that the last sent velocity command keeps on being executed.
        # Instead of letting the node die via signals (ROS node signals/Ctrl+C), we stop the robot manually.
        self.tf_listener = tf.TransformListener()
        self.fps = 10 # rospy.get_param("/align_task/freq")
        self.rate = rospy.Rate(self.fps)

        # Create the move_group object
        self.move_group = moveit_commander.MoveGroupCommander("robot")
        self.robot = moveit_commander.RobotCommander()
        #self.move_group.set_end_effector_link(self._end_effector_link)
        rospy.sleep(3)

        wrench_l = 100
        self.wrench_data = deque([WrenchStamped() for _ in range(wrench_l)], wrench_l)
        self.wrench_data_filtered = deque([WrenchStamped() for _ in range(wrench_l)], wrench_l)
        self.wrist_ft = [0.0]*6
        self.pose = None

        # Subscribers for observations
        if "wrist_ft" in self.observation_options:
            fttopic = "/wrench"
            rospy.Subscriber(fttopic, WrenchStamped, self.wrist_ft_callback)
            rospy.loginfo("Waiting for ft")
            rospy.wait_for_message(fttopic, WrenchStamped)
            rospy.loginfo("Connected to ft sensor.")

        if "wrist_ft_filtered" in self.observation_options:
            ft_topic_filtered = "/wrench/filtered"
            rospy.Subscriber(ft_topic_filtered, WrenchStamped, self.wrist_ft_filtered_callback)
            rospy.loginfo("Waiting for ft filtered")
            rospy.wait_for_message(ft_topic_filtered, WrenchStamped)
            rospy.loginfo("Connected to ft filtered.")

    def wrist_ft_callback(self, msg):
        self.wrench_data.append(msg)

    def wrist_ft_filtered_callback(self, msg):
        self.wrench_data_filtered.append(msg)

    def get_wrist_ft_averaged(self, number=15):
        if number == 0: # return last data
            data = self.wrench_data[-1]
            force = xyz_to_array(data.wrench.force)
            torque = xyz_to_array(data.wrench.torque)
            return force + torque

        fx, fy, fz = [], [], []
        mx, my, mz = [], [], []
        for i in range(number):
            wrench_msg = self.wrench_data[-(i+1)]
            fx.append(wrench_msg.wrench.force.x)
            fy.append(wrench_msg.wrench.force.y)
            fz.append(wrench_msg.wrench.force.z)
            mx.append(wrench_msg.wrench.torque.x)
            my.append(wrench_msg.wrench.torque.y)
            mz.append(wrench_msg.wrench.torque.z)
        fx = sum(fx)/len(fx)
        fy = sum(fy)/len(fy)
        fz = sum(fz)/len(fz)
        mx = sum(mx)/len(mx)
        my = sum(my)/len(my)
        mz = sum(mz)/len(mz)
        return [fx, fy, fz, mx, my, mz]

    def get_wrist_ft(self, averaged_number=0):
        return self.get_wrist_ft_averaged(number=averaged_number)

    def get_wrist_ft_filtered(self):
        data = self.wrench_data_filtered[-1]
        force = xyz_to_array(data.wrench.force)
        torque = xyz_to_array(data.wrench.torque)
        return force + torque

    def joint_state_callback(self, msg):
        self.joint_pos = list(msg.position)
        self.joint_vel = list(msg.velocity)

    def mocap_state_callback(self, msg):
        """
        Get the current mocap pose
        """
        time = rospy.get_time()
        # Orientation in Quaternion
        pos = msg.pose.position
        oq = msg.pose.orientation

        position = [pos.x, pos.y, pos.z]
        orientation = [oq.x, oq.y, oq.z, oq.w]
        pose = position + orientation
        self.mocap_state = pose

    def mocap_arm_tip_pos_callback(self, msg):
        """
        Get the current mocap pose
        """
        time = rospy.get_time()
        # Orientation in Quaternion
        pos = msg.pose.position
        oq = msg.pose.orientation

        position =    [pos.x, pos.y, pos.z]
        orientation = [oq.x, oq.y, oq.z, oq.w]
        pose = position + orientation
        self.mocap_arm_tip_pos_state = pose

    def get_cartesian_state(self):
        """
        Get the pose of the end effector. Pose is returned from the move group
        as a Pose message.
        """
        pose_msg = self.move_group.get_current_pose().pose
        pose = [pose_msg.position.x,
                pose_msg.position.y,
                pose_msg.position.z,
                pose_msg.orientation.x,
                pose_msg.orientation.y,
                pose_msg.orientation.z,
                pose_msg.orientation.w]

        return pose

    def stop_robot(self):
        self._publish_joint_velocity(self._zero_joints)

    def _publish_joint_velocity(self, joint_vel):
        """
        Publishes the commanded joint velocity to the controller.

        joint_vel (np.array) = the list of requested joint velocities
        """
        success = True
        # if self.check_wrist_overload():
        #   # flexwrist protection
        #   joint_vel = copy.deepcopy(self._zero_joints)
        #   success = False
        #print("success",joint_vel)
        joint_msg = Float64MultiArray() 
        joint_msg.layout.dim.append(MultiArrayDimension())
        joint_msg.layout.dim[0].label = '' #"joint velocity"
        joint_msg.layout.dim[0].size = 0 #6
        joint_msg.data = joint_vel
        self.joint_pub.publish(joint_msg)
        return success

    def switch_controllers(self, desired = "moveit"):
        """
        Switch between different controller types
        Allows switching between joint vel control and moveit planning,
        and disabling controllers for the gripper control
        """
        # The possible controller options
        joint_controller = 'joint_group_vel_controller'
        vel_controller = 'scaled_pos_joint_traj_controller'

        # Define the lists of controllers to start and stop for each method
        # "method name" : (to_start, to_stop)
        control_methods = {"moveit": ([vel_controller],  [joint_controller]),
                            "joint" : ([joint_controller],[vel_controller]),
                            "none"  : ([], [vel_controller,joint_controller])}

        # Wait for the controller manager
        # rospy.loginfo("Try to switch controller. Waiting for service")
        rospy.wait_for_service('/controller_manager/switch_controller')

        # Try to switch controllers
        try:
            sc = rospy.ServiceProxy('/controller_manager/switch_controller', 
                                    SwitchController)
            # Start and stop controllers according to the desired control method
            sc(start_controllers=control_methods[desired][0], stop_controllers=control_methods[desired][1], strictness=1)
            # rospy.loginfo("Switched from {} to {}".format(*control_methods[desired]))
        except rospy.ServiceException:
            rospy.logerr('Failed to switch controllers.')

    def go_to_cartesian_pose(self, pose, speed=1.0, end_effector_link=""):
        """
        Cartesian path planning
        """
        # Create pose
        pose_goal = Pose()
        pose_goal.position   .x = pose[0]
        pose_goal.position   .y = pose[1]
        pose_goal.position   .z = pose[2]
        pose_goal.orientation.x = pose[3]
        pose_goal.orientation.y = pose[4]
        pose_goal.orientation.z = pose[5]
        pose_goal.orientation.w = pose[6]

        if end_effector_link:
            # Change end effector temporary
            self.move_group.set_end_effector_link(end_effector_link)
            rospy.sleep(2)

        # if switch_controllers:
            # Switch to moveit
            # self.switch_controllers("moveit")
        waypoints = [pose_goal]
        # print(("pose_goal",waypoints))
        plan, fraction = self.move_group.compute_cartesian_path(waypoints, 0.01, 0.0)
        plan = self.move_group.retime_trajectory(self.robot.get_current_state(), plan, speed)

        self.move_group.execute(plan, wait=True)
        rospy.sleep(2)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        if end_effector_link:
            # Restore end effector link
            self.move_group.set_end_effector_link(self._end_effector_link)

        # if switch_controllers:
            # Switch back to joint control
            # self.switch_controllers("joint")
