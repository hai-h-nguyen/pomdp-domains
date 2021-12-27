import operator
from pathlib import Path
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import mujoco_py

class BoxEnv(gym.Env):

    def __init__(self, rendering=False, seed=None):
        """
        """

        super(BoxEnv, self).__init__()

        # X range
        self.x_left_limit = 0
        self.x_right_limit = 100
        self.x_g_left_limit = self.x_left_limit + 10
        self.x_g_right_limit = self.x_right_limit - 10

        self.go_to_left = 0.0

        # Boundary
        self.lboundary = 40
        self.rboundary = 60

        self.action_dim = 1

        #################### END CONFIGS #######################

        # mujoco-py
        xml_path = Path(__file__).resolve().parent / 'assets' / 'two_boxes.xml'
        self.model = mujoco_py.load_model_from_path(str(xml_path))
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None  # Initializes only when self.render() is called.
        self.rendering = rendering

        # Constants
        self.FINGER_TIP_OFFSET = 0.375

        # MuJoCo
        # bodies
        self.gripah_bid = self.model.body_name2id('gripah-base')
        self.small_box_bid = self.model.body_name2id('small_box')
        self.small_box_2_bid = self.model.body_name2id('small_box_2')

        self.big_box_bid = self.model.body_name2id('big_box')
        self.big_box_2_bid = self.model.body_name2id('big_box_2')

        self.lregion_bid = self.model.site_name2id('left_boundary')
        self.rregion_bid = self.model.site_name2id('right_boundary')        

        # geoms
        self.wide_finger_geom_id = self.model.geom_name2id('geom:wide-finger')
        self.wide_finger_tip_geom_id = self.model.geom_name2id('geom:wide-finger-tip')
        # joints
        self.slide_x_c_id = self.model.joint_name2id('slide:gripah-base-x')
        self.hinge_wide_finger_id = self.model.joint_name2id('hinge:wide-finger')
        self.hinge_narrow_finger_id = self.model.joint_name2id('hinge:narrow-finger')
        # actuators
        self.velocity_x_id = self.model.actuator_name2id('velocity:x')
        self.velocity_narrow_finger_id = self.model.actuator_name2id('velocity:narrow-finger')
        self.position_narrow_finger_id = self.model.actuator_name2id('position:narrow-finger')

        self.model.jnt_range[self.slide_x_c_id][0] = self.x_left_limit
        self.model.jnt_range[self.slide_x_c_id][1] = self.x_right_limit
        self._place_grid_marks()

        # Gripah
        self.default_velocity = 15
        self.step_length = 100
        self.low_stiffness = 200

        self.qpos_nfinger = 0

        self.x_box = None

        self.state_dim = 2
        self.low_obs_dim = 2

        self.prepare_high_obs_fn = lambda state: state

        # TODO: Tune this
        self.min_x_g_box_distance = 6

        # Action
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)

        self.x_g = 0
        self.theta = 0

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(2,), dtype=np.float32)

        # The finger is always soft
        self.model.jnt_stiffness[self.hinge_wide_finger_id] = self.low_stiffness

        # numpy random
        self.np_random = None
        self.seed(seed)

    def reset(self):
        """
        """

        # Resets the mujoco env
        self.sim.reset()

        self.box_1 = 40
        self.box_2 = 60

        self.x_g = 50

        # ok = False
        # while(not ok):
        #     self.x_g = self.np_random.uniform(self.x_g_left_limit, self.x_g_right_limit)
        #     if abs(self.x_g - self.box_1) >= self.min_x_g_box_distance and abs(self.x_g - self.box_2) >= self.min_x_g_box_distance:
        #         ok = True

        # Assigns the parameters to mujoco-py
        option = self.np_random.randint(4)

        # Both boxes are small
        if option == 0:
            # Pass this batch by going to the right
            self.go_to_left = False

            self.model.body_pos[self.small_box_bid][0] = self.box_1
            self.model.body_pos[self.small_box_2_bid][0] = self.box_2

            self.model.body_pos[self.big_box_bid][0] = -500
            self.model.body_pos[self.big_box_2_bid][0] = -500

        # Small box on the left, big box on the right, Not pass
        if option == 1:
            self.go_to_left = True

            self.model.body_pos[self.small_box_bid][0] = self.box_1
            self.model.body_pos[self.big_box_bid][0] = self.box_2

            self.model.body_pos[self.small_box_2_bid][0] = -500
            self.model.body_pos[self.big_box_2_bid][0] = -500

        # Small box on the right, big box on the left, Not pass
        if option == 2:
            self.go_to_left = True

            self.model.body_pos[self.small_box_bid][0] = self.box_2
            self.model.body_pos[self.big_box_bid][0] = self.box_1

            self.model.body_pos[self.small_box_2_bid][0] = -500
            self.model.body_pos[self.big_box_2_bid][0] = -500

        # Both boxes are big, Pass
        if option == 3:
            self.go_to_left = False

            self.model.body_pos[self.big_box_2_bid][0] = self.box_2
            self.model.body_pos[self.big_box_bid][0] = self.box_1

            self.model.body_pos[self.small_box_2_bid][0] = -500
            self.model.body_pos[self.small_box_bid][0] = -500

        # qpos
        self.sim.data.qpos[self.slide_x_c_id] = self.x_g + self.FINGER_TIP_OFFSET
        self._control_narrow_finger(theta_target=0.9, teleport=True)

        self._update_state()

        return self._get_obs()

    def step(self, action):
        """
        Steps the simulation with the given action and returns the observations.

        :param action: (movement)
        :return: the observations of the environment
        """

        env_reward = -1
        done = False

        self._move_gripper(action)

        self._update_state()

        if self.x_g <= self.x_g_left_limit:
            if self.go_to_left:
                env_reward = 0.0
            else:
                env_reward = -5.0 # negative reward

        if self.x_g >= self.x_g_right_limit:
            if not self.go_to_left:
                env_reward = 0.0
            else:
                env_reward = -5.0

        if self.x_g >= self.x_g_right_limit or self.x_g <= self.x_g_left_limit or env_reward >= 0.0:
            done = True

        return self._get_obs(), env_reward, done, {}

    def _get_obs(self):
        
        return np.array((self.x_g / self.x_right_limit, 
                self.theta))

    def render(self, mode='human'):
        if self.rendering:
            if self.viewer is None:
                self.viewer = mujoco_py.MjViewer(self.sim)
                self.viewer.cam.distance = 150
                self.viewer.cam.azimuth = 90
                self.viewer.cam.elevation = -15

            self.viewer.render()

    def close(self):
        pass

    def seed(self, seed=None):
        """
        Sets the seed for this environment's random number generator(s).

        :seed the seed for the random number generator(s)
        """

        self.np_random, seed_ = seeding.np_random(seed)

        return [seed_]

    def _update_state(self):
        """
        Samples the data from sensors and updates the state.
        """

        self.x_g = self._get_raw_x_g()
        self.theta = self._get_theta()

    def _move_gripper(self, movement):
        if self.x_g <= self.x_g_left_limit or self.x_g >= self.x_g_right_limit:
            return        

        self._control_slider_x(movement)

    def _control_slider_x(self, scale):
        """
        Controls the joint x of the gripah to move to the given target state.

        :param direction: scale
        """

        for _ in range(self.step_length):
            self.sim.data.ctrl[self.velocity_x_id] = scale * self.default_velocity
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
        Gets the current angle of the angle of the wide finger.

        :return: the current angle of the angle of the wide finger
        """
        return self._get_wide_finger_angle()

    def _get_wide_finger_angle(self):
        """
        Gets the current angle of the wide finger. Since the raw value is
        negative but a positive number is expected in this environment, the
        additive inverse of the result from the MuJoCo will be returned.

        :return: the current angle of the wide finger
        """

        return -self.sim.data.qpos[self.hinge_wide_finger_id]

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

        return self.sim.data.sensordata[3]

    def _get_gripah_raw_state(self):
        """
        Gets the current state of the gripah (x, y, z, and the angle of the narrow finger).

        :return: the current state of the gripah
        """

        x = self.sim.data.sensordata[0]
        y = self.sim.data.sensordata[1]
        z = self.sim.data.sensordata[2]
        w1 = self._get_wide_finger_angle()
        w2 = self._get_narrow_finger_angle()

        return x, y, z, w1, w2

    def _place_grid_marks(self):
        """
        Places all grid marks at the right positions.
        """

        grid_marker_0 = self.model.site_name2id('grid-marker-0')
        grid_marker_1 = self.model.site_name2id('grid-marker-1')
        grid_marker_2 = self.model.site_name2id('grid-marker-2')

        self.model.site_pos[grid_marker_0][0] = self.x_left_limit
        self.model.site_pos[grid_marker_1][0] = (self.x_left_limit + self.x_right_limit) / 2
        self.model.site_pos[grid_marker_2][0] = self.x_right_limit

        # place boundary marks
        self.model.site_pos[self.lregion_bid][0] = self.lboundary
        self.model.site_pos[self.rregion_bid][0] = self.rboundary
