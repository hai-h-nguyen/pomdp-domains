import gym
import gym_pomdps
from gym import spaces
import numpy as np

class HeavenHellEnv(gym.Env):

    def __init__(self, rendering=False):
        self.env = gym.make("POMDP-heavenhell-episodic-v0")
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        self.state = None
        self.belief = None

        self.NORTH = 0
        self.SOUTH = 1
        self.EAST  = 2
        self.WEST  = 3

    def close(self):
        pass

    def query_expert(self):
        if self.state in [7, 0, 1, 17, 10, 11]:
            return [self.NORTH]

        elif self.state in [12, 13, 14, 15, 16]:
            return [self.EAST]

        elif self.state in [8, 9, 18, 19]:
            return [self.WEST]

        elif self.state in [2, 3, 4, 5, 6]:
            return [self.WEST]

        else:
            raise NotImplementedError

    def _toOneHot(self, obs_idx):
        obs = np.zeros(11)
        obs[obs_idx] = 1.0
        return obs

    def seed(self, seed):
        self.env.seed(seed)

    def reset(self):
        self.state = self.env.reset_functional()
        self.belief = gym_pomdps.belief.belief_init(self.env)

        initial_action = np.random.randint(0, self.action_space.n)
        initial_obs, _, _, _ = self.step(initial_action)
        return initial_obs

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action[0]

        # Next state
        r = gym_pomdps.belief.expected_reward(self.env, self.belief, action)
        self.state, o, r, done, info = self.env.step_functional(self.state, action)

        # Update next belief
        self.belief = gym_pomdps.belief.belief_step(self.env, self.belief, action, o)

        if r in [1.0, -1.0]:
            done = True

        info["success"] = (r == 1.0)

        return self._toOneHot(o), r, done, info
