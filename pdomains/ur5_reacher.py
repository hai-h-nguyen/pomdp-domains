import time
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import gym
from gym import spaces
from pathlib import Path
from gym.utils import seeding
import gin

ASSETS_PATH = Path(__file__).resolve().parent / 'assets'

def ur5_bound_angle(angle):

    bounded_angle = np.absolute(angle) % (2*np.pi)
    if angle < 0:
        bounded_angle = -bounded_angle

    return bounded_angle

@gin.configurable
class UR5Env(gym.Env):

    def __init__(self, prepare_high_obs_method="final-selective", args=None, seed=None, num_frames_skip=15, rendering=False):
        #################### START CONFIGS #######################
        model_name = "ur5_reacher_no_wall.xml"

        initial_joint_pos = np.array([  5.96625837e-03,   3.22757851e-03,  -1.27944547e-01])
        initial_joint_pos = np.reshape(initial_joint_pos,(len(initial_joint_pos),1))
        initial_joint_ranges = np.concatenate((initial_joint_pos,initial_joint_pos),1)
        initial_joint_ranges[0] = np.array([-np.pi/8,np.pi/8])
        initial_state_space = np.concatenate((initial_joint_ranges,np.zeros((len(initial_joint_ranges),2))),0)

        # functions to project state to goal
        project_state_to_subgoal = lambda sim, state: \
                np.concatenate((np.array([ur5_bound_angle(sim.data.qpos[i]) for i in range(len(sim.data.qpos))]), \
                np.array([4 if sim.data.qvel[i] > 4 else -4 if sim.data.qvel[i] < -4 else sim.data.qvel[i] for i in range(len(sim.data.qvel))])))

        subgoal_bounds = np.array([[-2*np.pi, 2*np.pi],
                                   [-2*np.pi,2*np.pi],
                                   [-2*np.pi,2*np.pi],
                                   [-4,4],
                                   [-4,4],
                                   [-4,4]])

        velo_threshold = 2
        angle_threshold = np.deg2rad(10)
        subgoal_thresholds = np.concatenate((np.array([angle_threshold for i in range(3)]), \
                                             np.array([velo_threshold for i in range(3)])))

        self.end_goal_thresholds = 0.2

        # Configs for agent
        agent_params = {}
        agent_params["subgoal_test_perc"] = 0.3
        agent_params["random_action_perc"] = 0.2

        agent_params["atomic_noise"] = [0.1 for i in range(3)]
        agent_params["subgoal_noise"] = [0.03 for i in range(6)]

        agent_params["num_pre_training_episodes"] = -1

        agent_params["num_exploration_episodes"] = 50
        #################### END CONFIGS #######################
        self.agent_params = agent_params

        self.name = model_name

        MODEL_PATH = ASSETS_PATH / self.name

        # Create Mujoco Simulation
        self.model = load_model_from_path(str(MODEL_PATH))
        self.sim = MjSim(self.model)

        self.switch_threshold = 0.2

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        self.extra_dim = 6
        self.obs_dim = len(self.sim.data.qpos) + len(self.sim.data.qvel) + self.extra_dim # State will include (i) joint angles and coordinate of the target position
        self.action_dim = len(self.sim.model.actuator_ctrlrange) # low-level action dim
        self.action_bounds = self.sim.model.actuator_ctrlrange[:,1] # low-level action bounds

        self.action_offset = np.zeros((len(self.action_bounds))) # Assumes symmetric low-level action ranges
        self.subgoal_dim = len(subgoal_bounds)
        self.subgoal_bounds = subgoal_bounds

        self.goal_space_test = [[-np.pi, np.pi], [-np.pi/4, 0], [-np.pi/4, np.pi/4]]

        # Projection functions
        self.project_state_to_subgoal = project_state_to_subgoal
        self.project_state_to_end_goal = lambda sim, state: np.array([ur5_bound_angle(sim.data.qpos[i]) for i in range(len(sim.data.qpos))])

        # Convert subgoal bounds to symmetric bounds and offset.  Need these to properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        # End goal/subgoal thresholds
        self.subgoal_thresholds = subgoal_thresholds

        # Set inital state and goal state spaces
        self.initial_state_space = initial_state_space

        # Implement visualization if necessary
        self.visualize = rendering  # Visualization boolean
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = num_frames_skip

        # For Gym interface
        self.action_space = spaces.Box(
            low=np.array(self.sim.model.actuator_ctrlrange[:,0]),
            high=np.array(self.sim.model.actuator_ctrlrange[:,1]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32            
        )

        # self.switch_angles = [np.pi/2, -np.pi/2, np.pi/4]

        self.switch_pos = [0.5, 0.5, 0.5]

        self.near_switch_timestamp = -1
        self.steps_cnt = 0
        self.solved = False
        self.done = False

        self.target_visible_duration = 20 # number of timestep that the target is visible to the agent (after the switch is turned on)

        self.seed(seed)
        obs = self.reset()

        self.prepare_high_obs_method = prepare_high_obs_method

        print(prepare_high_obs_method)

        if prepare_high_obs_method in ['full', 'recurrent']:
            self.prepare_high_obs_fn = self.full_fn

        if prepare_high_obs_method in ['final']:
            self.prepare_high_obs_fn = self.final_fn

        if prepare_high_obs_method in ['final-selective']:
            self.prepare_high_obs_fn = self.final_selective_fn

        self.high_obs_dim = len(self.prepare_high_obs_fn(obs))
        self.low_obs_dim = len(self.prepare_low_obs_fn(obs))

        print(f"High obs dim {self.high_obs_dim}, Low obs dim {self.low_obs_dim}")

        self.max_ep_len = 200

    def full_fn(self, obs):
        return obs

    def final_selective_fn(self, obs):
        # get the final observation
        temp = obs[-self.obs_dim:]

        # only select the wrist position and the target position
        return temp[-6:]

    def final_fn(self, obs):
        return obs[-self.obs_dim:]

    def prepare_low_obs_fn(self, obs):
        return obs[:6]

    # Get state, which concatenates joint positions and velocities
    def _get_obs(self, target_pos, current_wrist_pos):
        return np.concatenate((self.sim.data.qpos, self.sim.data.qvel, current_wrist_pos, target_pos))

    # Reset simulation to state within initial state specified by user
    def reset(self):

        self.steps_cnt = 0
        self.near_switch_timestamp = -1
        self.solved = False
        self.done = False        

        goal_possible = False
        while not goal_possible:
            end_goal = np.zeros(shape=(3,))
            end_goal[0] = self.np_random.uniform(self.goal_space_test[0][0], self.goal_space_test[0][1])
            end_goal[1] = self.np_random.uniform(self.goal_space_test[1][0], self.goal_space_test[1][1])
            end_goal[2] = self.np_random.uniform(self.goal_space_test[2][0], self.goal_space_test[2][1])

            # Next need to ensure chosen joint angles result in achievable task (i.e., desired end effector position is above ground)

            theta_1 = end_goal[0]
            theta_2 = end_goal[1]
            theta_3 = end_goal[2]

            upper_arm_pos_2 = np.array([0,0.13585,0,1])
            forearm_pos_3 = np.array([0.425,0,0,1])
            wrist_1_pos_4 = np.array([0.39225,-0.1197,0,1])

            # Transformation matrix from shoulder to base reference frame
            T_1_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.089159],[0,0,0,1]])

            # Transformation matrix from upper arm to shoulder reference frame
            T_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],[np.sin(theta_1), np.cos(theta_1), 0, 0],[0,0,1,0],[0,0,0,1]])

            # Transformation matrix from forearm to upper arm reference frame
            T_3_2 = np.array([[np.cos(theta_2),0,np.sin(theta_2),0],[0,1,0,0.13585],[-np.sin(theta_2),0,np.cos(theta_2),0],[0,0,0,1]])

            # Transformation matrix from wrist 1 to forearm reference frame
            T_4_3 = np.array([[np.cos(theta_3),0,np.sin(theta_3),0.425],[0,1,0,0],[-np.sin(theta_3),0,np.cos(theta_3),0],[0,0,0,1]])

            forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
            wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

            # Make sure wrist 1 pos is above ground so can actually be reached
            if np.absolute(end_goal[0]) > np.pi/4 and forearm_pos[2] > 0.05 and wrist_1_pos[2] > 0.15:
                goal_possible = True

        self.target_pos = self._angles2jointpos(end_goal)[2]


        # Set initial joint positions and velocities
        for i in range(len(self.sim.data.qpos)):
            self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

        for i in range(len(self.sim.data.qvel)):
            self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

        self.sim.step()

        current_wrist_pos = self._angles2jointpos(self.sim.data.qpos)[2]

        # Not reveal the target info at reset
        return self._get_obs(np.zeros_like(self.target_pos), current_wrist_pos)

    # Execute low-level action for number of frames specified by num_frames_skip
    def step(self, action):

        self.steps_cnt += 1

        env_reward = -1

        if self.visualize:
            self.display_target_pos()
            self.display_joint_pos()
            self.display_switch()

        self.sim.data.ctrl[:] = action

        for _ in range(self.num_frames_skip):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        current_wrist_pos = self._angles2jointpos(self.sim.data.qpos)[2]

        dist2switch = np.linalg.norm(current_wrist_pos - self.switch_pos)

        near_switch = (dist2switch < self.switch_threshold)

        # Near the switch, record the timestamp
        if near_switch:
            self.near_switch_timestamp = self.steps_cnt

        # If the switch is turned on, let the agent see the target for a number of timesteps
        if self.near_switch_timestamp > 0 and (self.steps_cnt - self.near_switch_timestamp) <= self.target_visible_duration:
            target_pos = np.copy(self.target_pos)
        else:
            target_pos = np.zeros_like(self.target_pos)
            
        # Check if the gripper is within the goal achievement threshold

        dist2target = np.linalg.norm(current_wrist_pos - self.target_pos)

        # Having to turn on the light
        goal_achieved = (dist2target <= self.end_goal_thresholds) and (self.near_switch_timestamp > 0)

        reward = 0.0
        done = False

        # Calculate reward
        if goal_achieved:
            reward = 1.0
            env_reward = 0.0
            done = True
        else:
            env_reward = -1

        self.done = (reward > 0.0)
        self.solved = self.done

        return self._get_obs(target_pos, current_wrist_pos), env_reward, done, {"is_success": self.solved}

    # display the region around the switch
    def display_switch(self):
        self.sim.data.mocap_pos[1] = self.switch_pos 

    def display_subgoals(self, subgoals):
        angle = subgoals[0][:3] # only take the angles out
        joint_pos = self._angles2jointpos(angle)

        for j in range(3):
            self.sim.data.mocap_pos[2 + j] = joint_pos[j]

    def display_joint_pos(self):
        joint_pos = self._angles2jointpos(self.sim.data.qpos)

        for j in range(3):
            self.sim.data.mocap_pos[5 + j] = joint_pos[j]

    def display_target_pos(self):
        self.sim.data.mocap_pos[0] = self.target_pos

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def _angles2jointpos(self, angles):
        theta_1 = angles[0]
        theta_2 = angles[1]
        theta_3 = angles[2]

        upper_arm_pos_2 = np.array([0,0.13585,0,1])
        forearm_pos_3 = np.array([0.425,0,0,1])
        wrist_1_pos_4 = np.array([0.39225,-0.1197,0,1])

        # Transformation matrix from shoulder to base reference frame
        T_1_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.089159],[0,0,0,1]])

        # Transformation matrix from upper arm to shoulder reference frame
        T_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0], [np.sin(theta_1), np.cos(theta_1), 0, 0],[0,0,1,0],[0,0,0,1]])

        # Transformation matrix from forearm to upper arm reference frame
        T_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0],
                        [0, 1 ,0 , 0.13585], 
                        [-np.sin(theta_2),0,np.cos(theta_2),0],
                        [0,0,0,1]])

        # Transformation matrix from wrist 1 to forearm reference frame
        T_4_3 = np.array([[np.cos(theta_3), 0, np.sin(theta_3),0.425], 
                [0,1,0,0], 
                [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                [0,0,0,1]])

        # Determine joint position relative to original reference frame
        upper_arm_pos = T_1_0.dot(T_2_1).dot(upper_arm_pos_2)[:3]
        forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
        wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

        joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

        return joint_pos