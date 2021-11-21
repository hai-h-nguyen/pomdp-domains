import time
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import gym
from gym import spaces
from pathlib import Path
from gym.utils import seeding
import json

ASSETS_PATH = Path(__file__).resolve().parent / 'assets'

class AntTagEnv(gym.Env):

    def __init__(self, seed=None, num_frames_skip=15, rendering=False):

        model_name = "ant_tag_small.xml"

        initial_joint_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        initial_joint_pos = np.reshape(initial_joint_pos,(len(initial_joint_pos),1))
        initial_joint_ranges = np.concatenate((initial_joint_pos,initial_joint_pos),1)
        initial_joint_ranges[0] = np.array([-6,6])
        initial_joint_ranges[1] = np.array([-6,6])

        initial_state_space = np.concatenate((initial_joint_ranges,np.zeros((len(initial_joint_ranges)-1,2))),0)

        self.name = model_name

        MODEL_PATH = ASSETS_PATH / self.name

        # Create Mujoco Simulation
        self.model = load_model_from_path(str(MODEL_PATH))
        self.sim = MjSim(self.model)

        self.extra_dim = 2 # xy coordinates of the target

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        self.state_dim = len(self.sim.data.qpos) + len(self.sim.data.qvel) + self.extra_dim # State will include (i) joint angles and (ii) joint velocities, extra info
        self.action_dim = len(self.sim.model.actuator_ctrlrange) # low-level action dim

        # Set inital state and goal state spaces
        self.initial_state_space = initial_state_space

        self.cage_max_x = 4.5
        self.cage_max_y = 4.5

        # Implement visualization if necessary
        self.visualize = rendering  # Visualization boolean
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = num_frames_skip

        # For Gym interface
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.action_dim,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32            
        )

        self.visible_radius = 3.0
        self.tag_radius = 1.5
        self.min_distance = 5.0
        self.target_step = 0.5

        self.seed(seed)

    # Get state, which concatenates joint positions and velocities
    def _get_obs(self, target_pos_visible):
        if target_pos_visible:
            return np.concatenate((self.sim.data.qpos, self.sim.data.qvel, self.sim.data.mocap_pos[0][:2]))
        else:
            return np.concatenate((self.sim.data.qpos, self.sim.data.qvel, np.zeros(2)))

    # Reset simulation to state within initial state specified by user
    def reset(self):

        # Reset controls
        self.sim.data.ctrl[:] = 0

        # Set initial joint positions and velocities
        for i in range(len(self.sim.data.qpos)):
            self.sim.data.qpos[i] = self.np_random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

        for i in range(len(self.sim.data.qvel)):
            self.sim.data.qvel[i] = self.np_random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

        init_position_ok = False

        while (not init_position_ok):
            # Initialize the target's position
            target_pos = self.np_random.uniform(low=[-self.cage_max_x, -self.cage_max_y], high=[self.cage_max_x, self.cage_max_y])
            ant_pos = self.np_random.uniform(low=[-self.cage_max_x, -self.cage_max_y], high=[self.cage_max_x, self.cage_max_y])

            d2target = np.linalg.norm(ant_pos - target_pos)

            if d2target > self.min_distance:
                init_position_ok = True

        self.sim.data.mocap_pos[0][:2] = target_pos

        self.sim.data.qpos[:2] = ant_pos

        # Move 2 spheres along the ant
        self.sim.data.mocap_pos[1][:2] = ant_pos
        self.sim.data.mocap_pos[2][:2] = ant_pos

        self.sim.step()

        return self._get_obs(False)

    def _move_target(self, ant_pos, current_target_pos):
        target2ant_vec = ant_pos - current_target_pos
        target2ant_vec = target2ant_vec / np.linalg.norm(target2ant_vec)

        per_vec_1 = [target2ant_vec[1], -target2ant_vec[0]]
        per_vec_2 = [-target2ant_vec[1], target2ant_vec[0]]
        opposite_vec = -target2ant_vec

        vec_list = [per_vec_1, per_vec_2, opposite_vec, np.zeros(2)]

        chosen_vec_idx = self.np_random.choice(np.arange(4), p=[0.25, 0.25, 0.25, 0.25])

        chosen_vec = np.array(vec_list[chosen_vec_idx]) * self.target_step + current_target_pos

        if abs(chosen_vec[0]) > self.cage_max_x or abs(chosen_vec[1]) > self.cage_max_y:
            chosen_vec = current_target_pos

        self.sim.data.mocap_pos[0][:2] = chosen_vec

    def _do_reveal_target(self):

        ant_pos = self.sim.data.qpos[:2]
        target_pos = self.sim.data.mocap_pos[0][:2]

        d2target = np.linalg.norm(ant_pos - target_pos)
        if (d2target < self.visible_radius):
            reveal_target_pos = True
        else:
            reveal_target_pos = False

        return reveal_target_pos

    # Execute low-level action for number of frames specified by num_frames_skip
    def step(self, action):

        self.sim.data.ctrl[:] = action

        # TODO: Change the position of the target based on a fixed policy
        ant_pos = self.sim.data.qpos[:2]
        target_pos = self.sim.data.mocap_pos[0][:2]

        self._move_target(ant_pos, target_pos)

        # Move 2 spheres along the ant
        self.sim.data.mocap_pos[1][:2] = ant_pos
        self.sim.data.mocap_pos[2][:2] = ant_pos

        for _ in range(self.num_frames_skip):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        ant_pos = self.sim.data.qpos[:2]

        done = False
        env_reward = -1

        target_pos = self.sim.data.mocap_pos[0][:2]

        # + reward and terminate the episode if can tag the target
        d2target = np.linalg.norm(ant_pos - target_pos)
        if (d2target <= self.tag_radius):
            env_reward = 0
            done = True

        reveal_target_pos = self._do_reveal_target()

        return self._get_obs(reveal_target_pos), env_reward, done, {}


    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]
