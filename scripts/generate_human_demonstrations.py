from pdomains import *
import gym
import time
import numpy as np

import argparse
from robosuite.devices import Joystick
from pdomains.utils import EpisodeLogger


parser = argparse.ArgumentParser()
parser.add_argument("--pos-xy-scale", type=float, default=1.0, help="How much to scale position user inputs")
parser.add_argument("--pos-z-scale", type=float, default=0.75, help="How much to scale position user inputs")
parser.add_argument("--rot-scale", type=float, default=0.05, help="How much to scale rotation user inputs")
parser.add_argument("--start-index", type=int, default=0, help="Start index to name files")
parser.add_argument("--show-action", action="store_true", help="Visualize the action space of the robot")
parser.add_argument("--save-state", action="store_false", help="Save state files")
parser.add_argument("--domain", type=str, help="Domain to run", default="square-xyz")
parser.add_argument("--real", action="store_true", help="Running w/ real robot env")

args = parser.parse_args()

# initialize device
device = Joystick(pos_xy_scale=args.pos_xy_scale,
                    pos_z_scale=args.pos_z_scale,
                    rot_scale=args.rot_scale)

device.start_control()

if args.real:
    env = gym.make(f"peg-insertion-{args.domain}-real-v0")
else:
    env = gym.make(f"peg-insertion-{args.domain}-v0", rendering=True)

episode_cnt = 0

while True:
    step_cnt = 0
    true_step_cnt = 0

    time.sleep(0.1)

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
                print(f"Episode {episode_cnt + args.start_index} Success, Saved")

            print(step_cnt, true_step_cnt)
            break
