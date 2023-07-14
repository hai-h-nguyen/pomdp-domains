from pdomains import *
import gym
import numpy as np

import argparse
from pdomains.joystick import Joystick
from pdomains.utils import EpisodeLogger


parser = argparse.ArgumentParser()
parser.add_argument("--pos-xy-scale", type=float, default=1.0, help="How much to scale position user inputs")
parser.add_argument("--pos-z-scale", type=float, default=1.0, help="How much to scale position user inputs")
parser.add_argument("--rot-scale", type=float, default=0.05, help="How much to scale rotation user inputs")
parser.add_argument("--start-index", type=int, default=0, help="Start index to name files")
parser.add_argument("--show-action", action="store_true", help="Print the action")
parser.add_argument("--domain", type=str, help="Domain to run", default="round")

args = parser.parse_args()

# initialize device
device = Joystick(pos_xy_scale=args.pos_xy_scale,
                  pos_z_scale=args.pos_z_scale,
                  rot_scale=args.rot_scale)

device.start_control()

env = gym.make(f"peg-insertion-{args.domain}-real-xyz-v0")

episode_cnt = 0

while True:
    step_cnt = 0

    obs = env.reset()

    data_reader = EpisodeLogger(f"{args.start_index + episode_cnt}")
    episode_data = []

    while True:
        action_dict = device.get_controller_state()

        action = np.zeros(3)
        action[1] = action_dict["front_back"]
        action[0] = -action_dict["left_right"]
        action[2] = action_dict["up_down"]

        step_cnt += 1

        if args.show_action:
            print(action)

        if action_dict["reset"]:
            print(f"Episode failed, len {step_cnt}")
            break

        next_obs, reward, done, info = env.step(action)
        env.render()

        terminal = True if reward > 0 else done

        episode_data.append((obs, action, next_obs, reward, terminal))
        obs = next_obs.copy()

        if done or reward > 0:
            # only save successful episode
            if reward > 0:
                data_reader.log_episode(episode_data)
                episode_cnt += 1
                print(f"Saved episode {episode_cnt + args.start_index}, len {step_cnt}")
            break
