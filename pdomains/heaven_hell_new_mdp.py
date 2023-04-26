import gym
from gym import spaces
from gym.spaces import Discrete, Box
from mazelab import BaseEnv, Object, VonNeumannMotion, BaseMaze
from mazelab import DeepMindColor as color
import numpy as np


maze_layout = np.array([[1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 0, 1, 1, 1],
                        [1, 1, 1, 0, 1, 1, 1],
                        [1, 1, 1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)

obs_layout = np.array([[1, 1, 1, 1, 1, 1, 1],
                       [1, 4, 3, 2, 5, 6, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 0, 1, 1, 1],
                       [1, 1, 1, 7, 8, 9, 1],
                       [1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)


OBS_HEAVEN_LEFT = 9
OBS_HEAVEN_RIGHT = 10

# see https://github.com/abaisero/gym-pomdps/blob/master/gym_pomdps/pomdps/heavenhell.pomdp


class Maze(BaseMaze):
    def __init__(self):
        super().__init__()

    @property
    def size(self):
        return maze_layout.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(maze_layout == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(maze_layout == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, [])
        goal = Object('goal', 3, color.goal, False, [])
        hell = Object('hell', 4, color.lava, False, [])
        priest = Object('priest', 4, color.button, False, [])
        return free, obstacle, agent, goal, hell, priest


class CoreEnv(BaseEnv):
    def __init__(self):
        super().__init__()

        self.maze = Maze()
        self.motions = VonNeumannMotion()

        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

        self.obs_heaven_position = OBS_HEAVEN_LEFT

        self.visit_priest = False

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_heaven(self.maze.objects.agent.positions[0]):
            reward = +1
            done = True

        elif self._is_hell(self.maze.objects.agent.positions[0]):
            reward = -1
            done = True

        else:
            reward = 0
            done = False

        return self._value_to_obs(self.maze.objects.agent.positions), reward, done, {}

    def reset(self):
        self.maze.objects.agent.positions = [[3, 3]]
        self.maze.objects.priest.positions = [[4, 5]]

        self.visit_priest = False

        if np.random.rand() < 0.5:
            self.maze.objects.goal.positions = [[1, 1]]
            self.maze.objects.hell.positions = [[1, 5]]
            self.obs_heaven_position = OBS_HEAVEN_LEFT
        else:
            self.maze.objects.goal.positions = [[1, 5]]
            self.maze.objects.hell.positions = [[1, 1]]
            self.obs_heaven_position = OBS_HEAVEN_RIGHT

        return self._value_to_obs(self.maze.objects.agent.positions)

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_heaven(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def _is_hell(self, position):
        out = False
        for pos in self.maze.objects.hell.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()

    def _value_to_obs(self, position):
        obs_idx = obs_layout[position[0][0], position[0][1]]

        if obs_idx >= OBS_HEAVEN_LEFT:
            obs_idx = self.obs_heaven_position

            if not self.visit_priest:
                self.visit_priest = True

        if not self.visit_priest:
            heaven_direction = -1
        else:
            if self.obs_heaven_position == OBS_HEAVEN_LEFT:
                heaven_direction = 0
            else:
                heaven_direction = 1

        return self._toOneHot(obs_idx, heaven_direction)

    def _toOneHot(self, obs_idx, heaven_direction):
        obs = np.zeros(12)
        obs[obs_idx] = 1.0
        obs[-1] = heaven_direction
        return obs


class HeavenHellEnv(gym.Env):

    def __init__(self, rendering=False):
        self.core_env = CoreEnv()

        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)  # include heaven direction
        self.action_space = self.core_env.action_space

    def close(self):
        pass

    def render(self, mode='human'):
        self.core_env.render()

    def seed(self, seed):
        self.core_env.seed(seed)

    def reset(self):
        return self.core_env.reset()

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action[0]

        return self.core_env.step(action)
