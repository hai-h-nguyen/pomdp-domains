#
# Created by Xinchao Song on June 1, 2020.
#
import math
import operator
from pathlib import Path
from collections import namedtuple

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

import mujoco_py

# Theta discrete values
THETA_EXTRA_NEGATIVE = 0
THETA_NEGATIVE = 1
THETA_NEUTRAL = 2
THETA_POSITIVE = 3
THETA_EXTRA_POSITIVE = 4

# Action values
ACT_LEFT = 0
ACT_RIGHT = 1
ACT_BACKWARD = 2
ACT_FORWARD = 3
ACT_STAY = 4

# State namedtuple
State = namedtuple('State', 'x_g, y_g, theta, x_bump1, y_bump1, x_bump2, y_bump2')


class Bumps2DEnv(gym.Env):
    """
    Description:

    The simulation environment of a general two bump environment. Based on the PlateScriptSimulator.

    Observation:
         Type: Discrete
         Num    Observation         Values
         0      Gripah Position X   0, 1, 2, 3, 4
         1      Gripah Position Y   0, 1, 2, 3, 4
         2      Gripah Angle        0 (THETA_EXTRA_LEFT), 1 (THETA_LEFT), 2 (THETA_NEUTRAL), 3 (THETA_RIGHT),
                                    4 (THETA_EXTRA_RIGHT)
         3      Action              0 (ACT_LEFT), 1 (ACT_RIGHT), 2 (ACT_FORWARD), 3 (ACT_BACKWARD), 4 (ACT_STAY)
         4      Bump #1 Position X  1, 2, 3
         5      Bump #1 Position Y  1, 2, 3
         6      Bump #2 Position X  1, 2, 3
         7      Bump #2 Position Y  1, 2, 3

    Actions:
         Type: Discrete(3)
         Num    Action
         0      Moving left with soft finger
         1      Moving right with soft finger
         2      Moving forward with soft finger
         3      Moving backward with soft finger
         4      Stay

    Reward:
         Reward of 1 for successfully reach the larger bump (bump #1) after visit both bumps

     Starting State:
         The starting state of the gripah is assigned to x_g = random, y_g = random, and theta = THETA_NEUTRAL

     Episode Termination:
         when the gripah tried to grasp a bump.
     """

    def __init__(self, rendering=False, seed=None):
        """
        The initialization of the simulation environment.
        """

        # mujoco-py
        xml_path = Path(__file__).resolve().parent / 'mujoco_model_2' / 'bumps_2d_model.xml'
        self.model = mujoco_py.load_model_from_path(str(xml_path))
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None  # Initializes only when self.render() is called.
        self.rendering = rendering

        # MuJoCo
        self.gripah_body_id = self.model.body_name2id('gripah-base')
        self.bump1_body_id = self.model.body_name2id('bump1')
        self.bump2_body_id = self.model.body_name2id('bump2')
        self.slide_x_g_id = self.model.joint_name2id('slide:gripah-base-x')
        self.slide_y_g_id = self.model.joint_name2id('slide:gripah-base-y')
        self.hinge_g_id = self.model.joint_name2id('hinge:gripah-base')
        self.hinge_wide_finger_id = self.model.joint_name2id('hinge:wide-finger')
        self.hinge_narrow_finger_id = self.model.joint_name2id('hinge:narrow-finger')
        self.velocity_x_id = self.model.actuator_name2id('velocity:x')
        self.velocity_y_id = self.model.actuator_name2id('velocity:y')
        self.velocity_narrow_finger_id = self.model.actuator_name2id('velocity:narrow-finger')
        self.position_narrow_finger_id = self.model.actuator_name2id('position:narrow-finger')
        self.default_velocity = 10
        self.low_stiffness = 50
        self.qpos_nfinger = 0
        self.x_to_y_rotating_angle = -math.pi * 0.5

        # Gird
        self.x_origin = self.model.body_pos[self.gripah_body_id][0]
        self.y_origin = self.model.body_pos[self.gripah_body_id][1]
        self.grid_size = 4
        self.xy_low_limit = 0
        self.xy_high_limit = self.grid_size - 1
        self.size_scale = 16  # 1 unit size in the code means 8 unit in the MuJoCo
        self._place_grid_marks()

        # Gripah
        self.xy_g_low_limit = self.xy_low_limit
        self.xy_g_high_limit = self.xy_high_limit
        self.x_g = None
        self.y_g = None

        # Bumps
        # Bump #1 is larger one (reaching target) and bump #2 is the smaller one.
        self.xy_bump_low_limit = 0
        self.xy_bump_high_limit = self.xy_high_limit            

        self.x_bump1 = None
        self.y_bump1 = None
        self.x_bump2 = None
        self.y_bump2 = None
        self.is_bump1_touched = False  # Flag to check whether bumps #1 has been touched before.
        self.is_bump2_touched = False  # Flag to check whether bumps #2 has been touched before.

        # Theta
        self.FINGER_TO_X_NEG = THETA_EXTRA_NEGATIVE
        self.FINGER_TO_NEG = THETA_NEGATIVE
        self.FINGER_TO_DOWN = THETA_NEUTRAL
        self.FINGER_TO_POS = THETA_POSITIVE
        self.FINGER_TO_X_POS = THETA_EXTRA_POSITIVE
        self.max_theta = THETA_EXTRA_POSITIVE
        self.angle_str = {THETA_EXTRA_NEGATIVE: 't--', THETA_NEGATIVE: 't-',
                          THETA_NEUTRAL: 't0',
                          THETA_POSITIVE: 't+', THETA_EXTRA_POSITIVE: 't++'}
        self.theta = self.FINGER_TO_DOWN

        # Action
        self.action_str = {ACT_LEFT: 'MoveLeft', ACT_RIGHT: 'MoveRight',
                           ACT_FORWARD: 'MoveForward', ACT_BACKWARD: 'MoveBackward',
                           ACT_STAY: 'STAY'}
        self.action_space = spaces.Discrete(len(self.action_str))
        self.norm_action = self.action_space.n - 1

        self.steps_cnt = 0

        # States: (x_g, y_g, theta, x_bump1, y_bump1, x_bump2, y_bump2)

        # Obs: (x_g, y_g, theta, action, x_bump1, y_bump1, x_bump2, y_bump2)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(5,), dtype=np.float32)

        self.stay_cnt = 0

        # numpy random
        self.np_random = None
        self.seed(seed)

    def reset(self):
        """
        Resets the current MuJoCo simulation environment with given initial x coordinates of the two bumps.

        :return: the observations of the environment
        """

        # Resets the mujoco env
        self.sim.reset()

        self.steps_cnt = 0
        self.stay_cnt = 0

        # Determines the start state of x_bump1 and y_bump1.
        self.x_bump1 = self.np_random.randint(self.xy_bump_low_limit, self.xy_bump_high_limit + 1)
        self.y_bump1 = self.np_random.randint(self.xy_bump_low_limit, self.xy_bump_high_limit + 1)

        # Determines the start state of x_bump2 and y_bump2.
        while True:
            self.x_bump2 = self.np_random.randint(self.xy_bump_low_limit, self.xy_bump_high_limit + 1)
            self.y_bump2 = self.np_random.randint(self.xy_bump_low_limit, self.xy_bump_high_limit + 1)

            if ((self.x_bump2 - self.x_bump1) ** 2 + (self.y_bump2 - self.y_bump1) ** 2) > 1:
                break

        # Resets the bumps-touched flags.
        self.is_bump1_touched = False
        self.is_bump2_touched = False

        # Determines the start state of x_g and y_g.
        while True:
            self.x_g = self.np_random.randint(self.xy_g_low_limit, self.xy_g_high_limit + 1)
            self.y_g = self.np_random.randint(self.xy_g_low_limit, self.xy_g_high_limit + 1)

            # Bumps cannot be at the gripper positions
            cond1 = not ((self.x_bump1 == self.x_g) and (self.y_bump1 == self.y_g))
            cond2 = not ((self.x_bump2 == self.x_g) and (self.y_bump2 == self.y_g))

            if cond1 and cond2:
                break

        # Assigns the start state to mujoco-py
        self.model.body_pos[self.bump1_body_id][0] = self.x_bump1 * self.size_scale
        self.model.body_pos[self.bump1_body_id][1] = self.y_bump1 * self.size_scale
        self.model.body_pos[self.bump2_body_id][0] = self.x_bump2 * self.size_scale
        self.model.body_pos[self.bump2_body_id][1] = self.y_bump2 * self.size_scale
        self.sim.data.qpos[self.slide_x_g_id] = self.x_g * self.size_scale
        self.sim.data.qpos[self.slide_y_g_id] = self.y_g * self.size_scale
        self._control_narrow_finger(theta_target=0.9, teleport=True)
        self.theta = self._get_theta()

        return np.array((self.x_g / self.xy_g_high_limit, self.y_g / self.xy_g_high_limit,
                         self.theta / self.max_theta, 0.0, 0.0))

    def step(self, action):
        """
        Steps the simulation with the given action and returns the observations.

        :param action:
        0 - move left
        2 - move right
        3 - move backward
        4 - move forward
        5 - stay

        :return: the observations of the environment
        """

        reward = 0
        done = False

        # Executes the determined action.
        if action == ACT_LEFT:
            self._move_gripah_left()

        elif action == ACT_RIGHT:
            self._move_gripah_right()

        elif action == ACT_BACKWARD:
            self._move_gripah_backward()

        elif action == ACT_FORWARD:
            self._move_gripah_forward()

        elif action == ACT_STAY:
            self.stay_cnt += 1
            if self._check_success():
                reward = 1

            reward += self._get_penalty()

        else:
            raise ValueError("Unknown action index received.")

        self.steps_cnt += 1

        obs = np.array([self.x_g / self.xy_g_high_limit, self.y_g / self.xy_g_high_limit,
                        self.theta / self.max_theta, self.is_bump1_touched, self.is_bump2_touched])

        info = {}

        info['success'] = (reward == 1)
        done = (reward == 1) or (self.stay_cnt == 2)

        return obs, reward, done, info

    def query_expert(self):
        """
        Get action from an expert that sees the complete state
        , ie. reach the big bump directly
        """
        return self._go_and_stay(self.x_bump1, self.y_bump1)

    def _go_and_stay(self, desired_x, desired_y):
        """
        Return action that reach a desired (x, y) and stay at (x, y)
        """
        delta_x = desired_x - self.x_g
        delta_y = desired_y - self.y_g

        if delta_x > 0:
            return [ACT_RIGHT]
        elif delta_x < 0:
            return [ACT_LEFT]
        else:
            if delta_y > 0:
                return [ACT_FORWARD]
            elif delta_y < 0:
                return [ACT_BACKWARD]
            else:
                return [ACT_STAY]

    def query_state(self):
        """
        Gets the current state.

        :return: the current state
        """

        state_normalized = State(x_g=self.x_g / self.xy_g_high_limit,
                                 y_g=self.y_g / self.xy_g_high_limit,
                                 theta=self.theta / self.max_theta,
                                 x_bump1=self.x_bump1 / self.xy_bump_high_limit,
                                 y_bump1=self.y_bump1 / self.xy_bump_high_limit,
                                 x_bump2=self.x_bump2 / self.xy_bump_high_limit,
                                 y_bump2=self.y_bump2 / self.xy_bump_high_limit)

        return np.array(state_normalized)

    def render(self, mode='human'):
        """
        Renders the environment.
        """

        if mode != 'human':
            raise NotImplementedError("Only the human mode is supported.")

        if self.rendering:
            if self.viewer is None:
                self.viewer = mujoco_py.MjViewer(self.sim)
                self.viewer.cam.distance = 150
                self.viewer.cam.azimuth = 90
                self.viewer.cam.elevation = -15

            self.viewer.render()

    def close(self):
        """
        Closes the simulation environment.
        """

        # mujoco-py will close the env automatically.
        pass

    def seed(self, seed=None):
        """
        Sets the seed for this environment's random number generator(s).

        :seed the seed for the random number generator(s)
        """

        self.np_random, seed_ = seeding.np_random(seed)

        return [seed_]

    def _move_gripah_left(self):
        """
        Moves the gripah left with a soft finger in the allowed range.

        :return: True if the move succeeds, False otherwise
        """

        if self.x_g <= self.xy_g_low_limit:
            return

        self.x_g -= 1
        self._move_gripah_along_x(self.x_g, self.low_stiffness)

        # Checks if either bump has been touched.
        if not self.is_bump1_touched:
            self.is_bump1_touched = self.is_bump1_touched_now()

        if not self.is_bump2_touched:    
            self.is_bump2_touched = self.is_bump2_touched_now()

    def _move_gripah_right(self):
        """
        Moves the gripah right with a soft finger in the allowed range.

        :return: True if the move succeeds, False otherwise
        """

        if self.x_g >= self.xy_g_high_limit:
            return

        self.x_g += 1
        self._move_gripah_along_x(self.x_g, self.low_stiffness)

        # Checks if either bump has been touched.
        if not self.is_bump1_touched:
            self.is_bump1_touched = self.is_bump1_touched_now()

        if not self.is_bump2_touched:    
            self.is_bump2_touched = self.is_bump2_touched_now()

    def _move_gripah_backward(self):
        """
        Moves the gripah left with a soft finger in the allowed range.

        :return: True if the move succeeds, False otherwise
        """

        if self.y_g <= self.xy_g_low_limit:
            return

        self.y_g -= 1
        self._move_gripah_along_y(self.y_g, self.low_stiffness)

        # Checks if either bump has been touched.
        if not self.is_bump1_touched:
            self.is_bump1_touched = self.is_bump1_touched_now()

        if not self.is_bump2_touched: 
            self.is_bump2_touched = self.is_bump2_touched_now()

    def _move_gripah_forward(self):
        """
        Moves the gripah right with a soft finger in the allowed range.

        :return: True if the move succeeds, False otherwise
        """

        if self.y_g >= self.xy_g_high_limit:
            return

        self.y_g += 1
        self._move_gripah_along_y(self.y_g, self.low_stiffness)

        # Checks if either bump has been touched.
        if not self.is_bump1_touched:
            self.is_bump1_touched = self.is_bump1_touched_now()

        if not self.is_bump2_touched:    
            self.is_bump2_touched = self.is_bump2_touched_now()

    def _check_success(self):
        """
        :return: True if the the task is achieved
        """

        self.theta = self._get_theta()

        if (self.x_g == self.x_bump1 and self.y_g == self.y_bump1 \
            and self.is_bump1_touched and self.is_bump2_touched):
            return True

        return False

    def _get_penalty(self):
        if (self.x_g == self.x_bump1 and self.y_g == self.y_bump1):
            if not (self.is_bump1_touched and self.is_bump2_touched):
                return -100.0
            else:
                return 0.0
        return 0.0

    def _move_gripah_along_x(self, x_g, stiffness):
        """
        Moves the gripah with the given x_g and stiffness.
        :param x_g:       the target x_g for the gripah
        :param stiffness: the stiffness of the wide finger
        """

        self._set_wide_finger_stiffness(stiffness)
        self._control_slider_x(x_target=self.x_origin + x_g * self.size_scale)
        self.theta = self._get_theta()

    def _move_gripah_along_y(self, y_g, stiffness):
        """
        Moves the gripah with the given y_g and stiffness.
        :param y_g:       the target y_g for the gripah
        :param stiffness: the stiffness of the wide finger
        """

        self._set_wide_finger_stiffness(stiffness)
        self._control_slider_y(y_target=self.y_origin + y_g * self.size_scale)
        self.theta = self._get_theta()

    def _control_slider_x(self, x_target, teleport=False):
        """
        Controls the joint x of the gripah to move to the given target state.

        :param x_target: the target state that joint x of the gripah should move to.
        :param teleport:     teleport mode. The gripah will be teleported to the desired state without running
                             simulation. Note when running the actuator in teleport mode, the gripah is not able
                             to interact with other objects
        """

        if teleport:
            self.sim.data.qpos[self.hinge_g_id] = 0
            self.sim.data.qpos[self.slide_x_g_id] = x_target
            self.sim.data.ctrl[self.velocity_x_id] = 0
            self.sim.data.ctrl[self.velocity_y_id] = 0
            self.sim.step()

            return

        if self._get_raw_x_g() < x_target:
            while self._get_raw_x_g() <= x_target:
                self.sim.data.qpos[self.hinge_g_id] = 0
                self.sim.data.ctrl[self.velocity_x_id] = self.default_velocity
                self.sim.data.ctrl[self.velocity_y_id] = 0
                self.sim.step()
                self.render()

        elif self._get_raw_x_g() > x_target:
            while self._get_raw_x_g() >= x_target:
                self.sim.data.qpos[self.hinge_g_id] = 0
                self.sim.data.ctrl[self.velocity_x_id] = -self.default_velocity
                self.sim.data.ctrl[self.velocity_y_id] = 0
                self.sim.step()
                self.render()

    def _control_slider_y(self, y_target, teleport=False):
        """
        Controls the joint x of the gripah to move to the given target state.

        :param y_target: the target state that joint y of the gripah should move to.
        :param teleport:     teleport mode. The gripah will be teleported to the desired state without running
                             simulation. Note when running the actuator in teleport mode, the gripah is not able
                             to interact with other objects
        """

        if teleport:
            self.sim.data.qpos[self.hinge_g_id] = self.x_to_y_rotating_angle
            self.sim.data.qpos[self.slide_y_g_id] = y_target
            self.sim.data.ctrl[self.velocity_x_id] = 0
            self.sim.data.ctrl[self.velocity_y_id] = 0
            self.sim.step()

            return

        if self._get_raw_y_g() < y_target:
            while self._get_raw_y_g() <= y_target:
                self.sim.data.qpos[self.hinge_g_id] = self.x_to_y_rotating_angle
                self.sim.data.ctrl[self.velocity_x_id] = 0
                self.sim.data.ctrl[self.velocity_y_id] = self.default_velocity
                self.sim.step()
                self.render()

        elif self._get_raw_y_g() > y_target:
            while self._get_raw_y_g() >= y_target:
                self.sim.data.qpos[self.hinge_g_id] = self.x_to_y_rotating_angle
                self.sim.data.ctrl[self.velocity_x_id] = 0
                self.sim.data.ctrl[self.velocity_y_id] = -self.default_velocity
                self.sim.step()
                self.render()

    def _control_narrow_finger(self, theta_target, teleport=False):
        """
        Controls the narrow finger to rotate to the given target state.

        :param theta_target: the target state that the narrow finger should rotate to.
        :param teleport:     teleport mode. The gripah will be teleported to the desired state without running
                             simulation. Note when running the actuator in teleport mode, the gripah is not able
                             to interact with other objects
        """
        self.qpos_nfinger = -theta_target

        if teleport:
            self.sim.data.qpos[self.hinge_narrow_finger_id] = self.qpos_nfinger
            self.sim.data.ctrl[self.position_narrow_finger_id] = self.qpos_nfinger
            self.sim.step()

            return

        self.sim.data.ctrl[self.position_narrow_finger_id] = self.qpos_nfinger
        while True:
            last_state = self._get_gripah_raw_state()
            self.sim.step()
            self.render()
            now_state = self._get_gripah_raw_state()

            for diff in map(operator.sub, last_state, now_state):
                if abs(round(diff, 3)) > 0.001:
                    break
            else:
                break

    def _get_theta(self):
        """
        Gets the current angle of the angle of the wide finger

        :return: the current angle of the angle of the wide finger
        """

        theta = self._get_wide_finger_angle()

        if theta < -0.2:
            theta_discretized = self.FINGER_TO_X_NEG
        elif theta < -0.08:
            theta_discretized = self.FINGER_TO_NEG
        elif theta > 0.2:
            theta_discretized = self.FINGER_TO_X_POS
        elif theta > 0.1:
            theta_discretized = self.FINGER_TO_POS
        else:
            theta_discretized = self.FINGER_TO_DOWN

        return theta_discretized

    def _get_wide_finger_angle(self):
        """
        Gets the current angle of the wide finger. Since the raw value is
        negative but a positive number is expected in this environment, the
        additive inverse of the result from the MuJoCo will be returned.

        :return: the current angle of the wide finger
        """

        return -self.sim.data.qpos[self.hinge_wide_finger_id]

    def _get_wide_finger_stiffness(self):
        """
        Gets the current stiffness of the wide finger.

        :return: the current stiffness of the wide finger
        """

        return self.model.jnt_stiffness[self.hinge_wide_finger_id]

    def _set_wide_finger_stiffness(self, stiffness):
        """
        Sets the current stiffness of the wide finger.

        :param stiffness the stiffness to be set
        """

        self.model.jnt_stiffness[self.hinge_wide_finger_id] = stiffness

    def _get_narrow_finger_angle(self):
        """
        Gets the current angle of the narrow finger. Since the raw value is
        negative but a positive number is expected in this environment, the
        additive inverse of the result from the MuJoCo will be returned.

        :return: the current angle of the narrow finger
        """

        return -self.sim.data.qpos[self.hinge_narrow_finger_id]

    def _get_narrow_finger_stiffness(self):
        """
        Gets the current stiffness of the narrow finger.

        :return: the current stiffness of the narrow finger
        """

        return self.model.model.jnt_stiffness[self.hinge_narrow_finger_id]

    def _get_raw_x_g(self):
        """
        Gets the raw value of x_g in MuJoCo.

        :return: the raw value of x_g
        """

        return self.sim.data.sensordata[0]

    def _get_raw_y_g(self):
        """
        Gets the raw value of x_g in MuJoCo.

        :return: the raw value of x_g
        """

        return self.sim.data.sensordata[1]

    def _get_gripah_raw_state(self):
        """
        Gets the current state of the gripah (x, y, z, and the angle of the narrow finger).

        :return: the current state of the gripah
        """

        x = self.sim.data.sensordata[0]
        y = self.sim.data.sensordata[1]
        z = self.sim.data.sensordata[2]
        w = self._get_narrow_finger_angle()

        return x, y, z, w

    def _place_grid_marks(self):
        """
        Places all grid marks at the right positions.
        """

        for _x in range(self.grid_size):
            for _y in range(self.grid_size):
                grid_mark_id = self.model.site_name2id('grid-mark-%d-%d' % (_x, _y))
                self.model.site_pos[grid_mark_id][0] = _x * self.size_scale
                self.model.site_pos[grid_mark_id][1] = _y * self.size_scale

    def is_bump1_touched_now(self):
        """
        Determines if bump #1 is being touched now.

        :return: True if bump #1 is being touched now, False otherwise
        """

        return self.x_g == self.x_bump1 and self.y_g == self.y_bump1

    def is_bump2_touched_now(self):
        """
        Determines if bump #1 is being touched now.

        :return: True if bump #1 is being touched now, False otherwise
        """

        return self.x_g == self.x_bump2 and self.y_g == self.y_bump2