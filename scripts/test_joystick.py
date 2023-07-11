from pdomains import *
import gym
import time
import numpy as np

import argparse
from pdomains.joystick import Joystick
from pdomains.utils import EpisodeLogger

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--pos-xy-scale", type=float, default=1.0, help="How much to scale position user inputs")
parser.add_argument("--pos-z-scale", type=float, default=1.0, help="How much to scale position user inputs")
parser.add_argument("--rot-scale", type=float, default=0.05, help="How much to scale rotation user inputs")
parser.add_argument("--start-index", type=int, default=0, help="Start index to name files")
parser.add_argument("--show-action", action="store_true", help="Visualize the action space of the robot")
parser.add_argument("--save-state", action="store_false", help="Save state files")
parser.add_argument("--domain", type=str, help="Domain to run", default="square-xyz")

args = parser.parse_args()

# initialize device
device = Joystick(pos_xy_scale=args.pos_xy_scale,
                    pos_z_scale=args.pos_z_scale,
                    rot_scale=args.rot_scale)

env = gym.make("peg-insertion-round-real-xyz-v0")

while True:
    obs = env.reset()

    device.start_control()

    while True:
        action_dict = device.get_controller_state()

        action = np.zeros(3)
        action[1] = action_dict["front_back"]
        action[0] = -action_dict["left_right"]
        action[2] = action_dict["up_down"]

        reset = action_dict["reset"]

        if reset:
            break

        obs, reward, done, info = env.step(action)

        if reward > 0:
            print("Suceed!")
            break