# -*- coding: utf-8 -*-

from more_itertools import first
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from helping_hands_rl_envs import env_factory
import matplotlib.pyplot as plt
import time
import matplotlib.patches as patches
import math
import torch

class BlockEnv(gym.Env):
    def __init__(self, seed=0, img_size=84, rendering=False, robot='kuka', action_sequence='pxyzr', noise=False):

        workspace = np.asarray([[0.3, 0.7],
                                [-0.2, 0.2],
                                [0.01, 0.25]])

        self.image_size = img_size
        # in RAD envs, image_size is greater than true_image_size
        self.true_image_size = 84

        self.env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': self.image_size, 'render': False, 'fast_mode': True,
                        'seed': seed, 'action_sequence': action_sequence, 'num_objects': 1, 'random_orientation': False,
                        'reward_type': 'sparse', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': robot,
                        'workspace_check': 'point', 'physics_mode': 'fast', 'hard_reset_freq': 1000, 'view_scale': 1.0,
                        'object_scale_range': (1, 1), 'obs_type': 'pixel',
                        'view_type': 'camera_center_xyz'}

        self.planner_config = {'random_orientation': False, 'dpos': 0.05, 'drot': np.pi/8}

        self.xyz_range = self.planner_config['dpos']
        self.r_range = self.planner_config['drot']
        
        self.env_config['render'] = rendering
        self.seed(seed)
        self.core_env = env_factory.createSingleProcessEnv('close_loop_pomdp_block_pulling',
                                                            self.env_config,
                                                            self.planner_config)

        self.viewer = None
        self.show = rendering
        self.obs = None
        self.include_noise = noise

        self.action_dim = len(self.env_config['action_sequence'])
        high_action = np.ones(self.action_dim)
        self.action_space = spaces.Box(-high_action, high_action)

        num_channels = 3

        low = np.zeros((num_channels, self.true_image_size, self.true_image_size))
        high = np.ones((num_channels, self.true_image_size, self.true_image_size))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.img_size = (num_channels, self.true_image_size, self.true_image_size)
        self.image_space = gym.spaces.Box(
            shape=self.img_size, low=0, high=1.0, dtype=np.float32
        )
        # self.observation_space = gym.spaces.Box(
        #     shape=(np.array(self.img_size).prod(),), low=0, high=1.0, dtype=np.float32
        # )

        self.target_obj_idx = 0

        self.step_cnt = 0

        self.old_obs = None

    def query_expert(self, episode_idx):
        """_summary_

        Args:
            episode_idx (int): used to choose whether to pick the movable/immovable block first

        Returns:
            _type_: expert action
        """
        if episode_idx % 2 == 1:
            return self.pull_movable()
        else:
            if self.step_cnt <= 10:
                return self.pull_immovable()
            elif self.step_cnt <= 12:
                self.signal_reset_target()
                return self.move_up()
            else:
                return self.pull_movable()

    def signal_reset_target(self):
        self.core_env.getNextAction(2)

    def pull_movable(self):
        """pull the movable block"""
        action = self.core_env.getNextAction(1)
        action[1:4] /= self.xyz_range

        if self.action_dim == 5:
            action[4] /= self.r_range

        if self.env_config['robot'] == 'kuka':
            action[0] = 2*action[0] - 1

        return action

    def pull_immovable(self):
        """pull the immovable block"""
        action = self.core_env.getNextAction(0)
        action[1:4] /= self.xyz_range

        if self.action_dim == 5:
            action[4] /= self.r_range

        if self.env_config['robot'] == 'kuka':
            action[0] = 2*action[0] - 1

        return action

    # def query_expert(self):
    #     """pick the movable block"""
    #     action = self.core_env.getNextAction(0)
    #     action[1:4] /= self.xyz_range

    #     if self.action_dim == 5:
    #         action[4] /= self.r_range

    #     if self.env_config['robot'] == 'kuka':
    #         action[0] = 2*action[0] - 1

    #     return action

    def move_up(self):
        """pick the immovable block"""
        action = self.core_env.getNextAction(1 - self.target_obj_idx)
        action[1:4] /= self.xyz_range

        action[1] = 0.0
        action[2] = 0.0
        action[3] = 1.0

        if self.action_dim == 5:
            action[4] /= self.r_range

        if self.env_config['robot'] == 'kuka':
            action[0] = 2*action[0] - 1

        return action

    def seed(self, seed=0):
        self.env_config['seed'] = seed
        self.np_random, seed_ = seeding.np_random(seed)
        return seed_

    @staticmethod
    def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])

        grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1) % 1
        angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

        tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]] \
            .repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)

        dot = lambda grad, shift: (
                torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

        n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
        t = fade(grid[:shape[0], :shape[1]])

        return (math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1]))

    # def _process_obs(self, state, obs):
    #     if self.include_noise:
    #         obs[0] += 0.007*self.rand_perlin_2d((self.image_size, self.image_size), (
    #             (np.random.choice([1, 2, 4, 6], 1)[0]),
    #             int(np.random.choice([1, 2, 4, 6], 1)[0]))).numpy()

    #     state_tile = state*np.ones((1, obs.shape[1], obs.shape[2]))
    #     stacked = np.concatenate([obs, state_tile], axis=0)
    #     return stacked

    def _process_obs(self, state, obs, seg_mask):
        if self.include_noise:
            obs[0] += 0.007*self.rand_perlin_2d((self.image_size, self.image_size), (
                (np.random.choice([1, 2, 4, 6], 1)[0]),
                int(np.random.choice([1, 2, 4, 6], 1)[0]))).numpy()

        state_tile = state*np.ones((1, obs.shape[1], obs.shape[2]))

        obj_ids = np.unique(seg_mask)

        for idx in obj_ids:
            # ignore irrelevant objects
            if idx not in [0, 2, 3]:
                seg_mask[seg_mask == idx] = 0

            # object 2 and 3 are the same type of object
            if idx in [2, 3]:
                seg_mask[seg_mask == idx] = 1

        # plt.imshow(seg_mask)
        # plt.show()

        seg_mask = seg_mask[None, :, :]

        # obj_ids = np.unique(seg_mask)

        # num_objs = 0

        # if 2 in obj_ids:
            # num_objs += 1

        # if 3 in obj_ids:
            # num_objs += 1

        # object 2 and 3 are the same type of object
        # if 2 in obj_ids and 3 in obj_ids:
            # num_objs -= 1

        # assert 0 <= num_objs <= 2, obj_ids

        # objs_mask = num_objs*np.ones((1, obs.shape[1], obs.shape[2]))
        stacked = np.concatenate([obs, state_tile, seg_mask], axis=0)
        return stacked

    def step(self, action):
        action[1:4] *= self.xyz_range  # scale from [-1, 1] to [-0.05, 0.05] for xyz

        if self.action_dim == 5:  # scale from [-1, 1] to [-np.pi/8, np.pi/8]
            action[4] *= self.r_range

        if self.env_config['robot'] == 'kuka':
            action[0] = 0.5 * (action[0] + 1)  # [-1, 1] to [0, 1] for p
        (state, seg_mask, obs), reward, done = self.core_env.step(action)

        # boxes = self.compute_boxes(seg_mask)

        # self.show_seg(obs[0], boxes)

        # plt.imshow(seg_mask)
        # plt.show()

        self.obs = self._process_obs(state, obs, seg_mask)

        info = {}

        info["success"] = done and (reward > 0)

        self.step_cnt += 1

        if self.show:
            self.render()

        return self.obs, reward, done, info

    def render(self, mode='human'):
        pass

    def reset(self):
        self.target_obj_idx = 1 - self.target_obj_idx
        self.step_cnt = 0
        (state, seg_mask, obs) = self.core_env.reset(self.target_obj_idx, noise=self.include_noise)
        self.obs = self._process_obs(state, obs, seg_mask)

        # boxes = self.compute_boxes(seg_mask)

        # self.show_seg(obs[0], boxes)

        # if self.old_obs is not None:
        #     diff = obs[0] - self.old_obs
        #     print(np.min(diff), np.max(diff), np.mean(diff))

        # self.old_obs = obs

        return self.obs

    def close(self):
        self.core_env.close()

    def masks_to_boxes(self, masks: np.array) -> np.array:
        """
        Compute the bounding boxes around the provided masks.

        Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
        ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

        Args:
            masks (Array[N, H, W]): masks to transform where N is the number of masks
                and (H, W) are the spatial dimensions.

        Returns:
            Array[N, 4]: bounding boxes
        """
        n = masks.shape[0]

        bounding_boxes = np.zeros((n, 4))

        for index, mask in enumerate(masks):
            y, x = np.where(mask != 0)

            bounding_boxes[index, 0] = np.min(x)
            bounding_boxes[index, 1] = np.min(y)
            bounding_boxes[index, 2] = np.max(x)
            bounding_boxes[index, 3] = np.max(y)

        return bounding_boxes

    def show_seg(self, obs, boxes):
        fig, ax = plt.subplots(1)

        ax.imshow(obs)

        for i, bbox in enumerate(boxes):
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            rect = patches.Rectangle((bbox[0], bbox[1]), width, height, linewidth=1,
                                    edgecolor='r', facecolor='none')

            ax.add_patch(rect)

        plt.show()

    def compute_boxes(self, obs):
        obj_ids = np.unique(obs)

        obj_ids = obj_ids[1:]

        ids = []
        if 2 in obj_ids:
            ids.append(2)

        if 3 in obj_ids:
            ids.append(3)

        obj_ids = np.array(ids)

        # remove background and noise

        # print(obj_ids)
        # np.delete(obj_ids, [1])

        # print(obj_ids)

        # print("----------")

        masks = obs == obj_ids[:, None, None]

        boxes = self.masks_to_boxes(masks)

        return boxes
