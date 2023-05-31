from pdomains import *
import matplotlib.pyplot as plt
import gym
import time
import numpy as np
import csv

import argparse
from robosuite.devices import Joystick


class EpisodeLogger:
    def __init__(self, filename):
        self.filename = filename

    def log_episode(self, episode):
        valid_episode = []
        for i in range(len(episode)):
            obs, action, next_obs, reward, done, true_step_cnt = episode[i]
            valid_episode.append((obs, action, next_obs, reward, done))
        if valid_episode:
            assert len(valid_episode) == true_step_cnt, f"{len(valid_episode)} != {true_step_cnt}"
            obs, action, next_obs, reward, done = zip(*valid_episode)
            np.savez(self.filename, obs=np.array(obs), action=np.array(action), 
                     next_obs=np.array(next_obs), reward=np.array(reward), done=np.array(done))

parser = argparse.ArgumentParser()
parser.add_argument("--pos-xy-scale", type=float, default=1.0, help="How much to scale position user inputs")
parser.add_argument("--pos-z-scale", type=float, default=1.0, help="How much to scale position user inputs")
parser.add_argument("--rot-scale", type=float, default=0.05, help="How much to scale rotation user inputs")
parser.add_argument("--show-action", action="store_true", help="Visualize the action space of the robot")
args = parser.parse_args()

# initialize device
device = Joystick(pos_xy_scale=args.pos_xy_scale,
                    pos_z_scale=args.pos_z_scale,
                    rot_scale=args.rot_scale)

device.start_control()

env = gym.make('pdomains-peg-insertion-v0', rendering=True)

episode_cnt = 0

while True:
    step_cnt = 0
    true_step_cnt = 0
    obs = env.reset()

    data_reader = EpisodeLogger(f"{episode_cnt}")
    episode_data = []
    
    while True:
        action_dict = device.get_controller_state()

        action = np.zeros(7)
        action[0] = action_dict["front_back"]
        action[1] = action_dict["left_right"]
        action[2] = action_dict["up_down"]
        # action[3:6] = action_dict["rot_left_right"]
        # action[6] = -1  # gripper no control

        step_cnt += 1

        if args.show_action:
            print(action)

        if action_dict["reset"]:
            print("Episode Failed, Not Saved")
            print(step_cnt, true_step_cnt)
            break

        next_obs, reward, done, info = env.step(action)
        env.render()

        terminal = True if reward > 0 else done

        if np.linalg.norm(action) > 0 or reward > 0:
            true_step_cnt += 1
            episode_data.append((obs, action[:3], next_obs, reward, terminal, true_step_cnt))
            obs = next_obs.copy()

        if done or reward > 0:
            if reward > 0:
                data_reader.log_episode(episode_data)
                episode_cnt += 1
                print("Episode Success, Saved", np.linalg.norm(action))
            else:
                print("Episode Failed, Not Saved")

            print(step_cnt, true_step_cnt)
            break