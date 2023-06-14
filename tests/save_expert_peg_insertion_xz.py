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
            obs, action, next_obs, reward, done, true_step_cnt, \
            state, next_state, full_obs, next_full_obs = episode[i]
            valid_episode.append((obs, action, next_obs, reward, done, state, next_state, full_obs, next_full_obs))
        if valid_episode:
            assert len(valid_episode) == true_step_cnt, f"{len(valid_episode)} != {true_step_cnt}"
            obs, action, next_obs, reward, done, state, next_state, full_obs, next_full_obs = zip(*valid_episode)
            np.savez(self.filename, obs=np.array(obs), action=np.array(action),
                     next_obs=np.array(next_obs), reward=np.array(reward), done=np.array(done))

            np.savez('state-' + self.filename, obs=np.array(state), action=np.array(action),
                     next_obs=np.array(next_state), reward=np.array(reward), done=np.array(done))

            np.savez('full-obs-' + self.filename, obs=np.array(full_obs), action=np.array(action),
                     next_obs=np.array(next_full_obs), reward=np.array(reward), done=np.array(done))


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

env = gym.make('peg-insertion-square-xz-v0', rendering=True)

episode_cnt = 0
start_episode = episode_cnt

while True:
    step_cnt = 0
    true_step_cnt = 0

    time.sleep(0.5)

    obs = env.reset()
    state = env.get_state()
    full_obs = env.get_full_obs()

    data_reader = EpisodeLogger(f"{episode_cnt}")
    episode_data = []

    while True:
        action_dict = device.get_controller_state()

        action = np.zeros(2)
        action[0] = action_dict["front_back"]
        # action[1] = action_dict["left_right"]
        action[1] = action_dict["up_down"]

        step_cnt += 1

        if args.show_action:
            print(action)

        if action_dict["reset"]:
            print("Episode Failed, Not Saved")
            print(step_cnt, true_step_cnt)
            break

        next_obs, reward, done, info = env.step(action)
        next_state = env.get_state()
        next_full_obs = env.get_full_obs()
        env.render()

        terminal = True if reward > 0 else done

        # only buffer the data if action is non-zero or reward is positive
        if np.linalg.norm(action) > 0 or reward > 0:
            true_step_cnt += 1
            episode_data.append((obs, action, next_obs, reward, terminal,
                                true_step_cnt, state, next_state, full_obs, next_full_obs))
            obs = next_obs.copy()
            state = next_state.copy()
            full_obs = next_full_obs.copy()

        if done or reward > 0:
            # only save successful episode
            if reward > 0:
                data_reader.log_episode(episode_data)
                episode_cnt += 1
                print(f"Episode {episode_cnt - start_episode} Success, Saved")
            else:
                print("Episode Failed, Not Saved")

            print(step_cnt, true_step_cnt)
            break
