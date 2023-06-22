from pdomains import *
import matplotlib.pyplot as plt
import gym
import time
import numpy as np

import argparse
from robosuite.devices import Joystick


class EpisodeLogger:
    def __init__(self, filename):
        self.filename = filename

    def log_episode(self, episode, save_state):
        valid_episode = []
        for i in range(len(episode)):
            obs, action, next_obs, reward, done, true_step_cnt, \
            state, next_state = episode[i]
            valid_episode.append((obs, action, next_obs, reward, done, state, next_state))

        if valid_episode:
            assert len(valid_episode) == true_step_cnt, f"{len(valid_episode)} != {true_step_cnt}"

            obs, action, next_obs, reward, done, state, next_state = zip(*valid_episode)

            np.savez(self.filename, obs=np.array(obs), action=np.array(action),
                     next_obs=np.array(next_obs), reward=np.array(reward), done=np.array(done))

            if save_state:
                np.savez('state-' + self.filename, obs=np.array(state), action=np.array(action),
                        next_obs=np.array(next_state), reward=np.array(reward), done=np.array(done))

parser = argparse.ArgumentParser()
parser.add_argument("--pos-xy-scale", type=float, default=1.0, help="How much to scale position user inputs")
parser.add_argument("--pos-z-scale", type=float, default=1.0, help="How much to scale position user inputs")
parser.add_argument("--rot-scale", type=float, default=0.05, help="How much to scale rotation user inputs")
parser.add_argument("--start-index", type=int, default=0, help="Start index to name files")
parser.add_argument("--show-action", action="store_true", help="Visualize the action space of the robot")
parser.add_argument("--save-state", action="store_true", help="Save state files")
parser.add_argument("--domain", type=str, help="Domain to run", default="square-state-xyz")

args = parser.parse_args()

# initialize device
device = Joystick(pos_xy_scale=args.pos_xy_scale,
                    pos_z_scale=args.pos_z_scale,
                    rot_scale=args.rot_scale)

device.start_control()

env = gym.make(f"peg-insertion-{args.domain}-v0", rendering=True)

episode_cnt = 0
start_episode = episode_cnt

while True:
    step_cnt = 0
    true_step_cnt = 0

    time.sleep(0.5)

    obs = env.reset()
    state = env.get_state()

    data_reader = EpisodeLogger(f"{args.start_index + episode_cnt}")
    episode_data = []

    while True:
        action_dict = device.get_controller_state()

        if env.action_space.shape[0] == 3:
            action = np.zeros(3)
            action[0] = action_dict["front_back"]
            action[1] = action_dict["left_right"]
            action[2] = action_dict["up_down"]
        elif env.action_space.shape[0] == 2:
            action = np.zeros(2)
            action[0] = action_dict["front_back"]
            action[1] = action_dict["up_down"]
        else:
            raise NotImplementedError

        step_cnt += 1

        if args.show_action:
            print(action)

        if action_dict["reset"]:
            print("Episode Failed, Not Saved")
            print(step_cnt, true_step_cnt)
            break

        next_obs, reward, done, info = env.step(action)
        next_state = env.get_state()
        env.render()

        terminal = True if reward > 0 else done

        # only buffer the data if action is non-zero or reward is positive
        if np.linalg.norm(action) > 0 or reward > 0:
            true_step_cnt += 1
            episode_data.append((obs, action, next_obs, reward, terminal,
                                 true_step_cnt, state, next_state))
            obs = next_obs.copy()
            state = next_state.copy()

        if done or reward > 0:
            # only save successful episode
            if reward > 0:
                data_reader.log_episode(episode_data, args.save_state)
                episode_cnt += 1
                print(f"Episode {episode_cnt - start_episode + args.start_index} Success, Saved")
            else:
                print("Episode Failed, Not Saved")

            print(step_cnt, true_step_cnt)
            break
