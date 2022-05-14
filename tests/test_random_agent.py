from pdomains import *
import gym
import numpy as np

env=gym.make('pdomains-mg-memory-s17-v0', rendering=False)
env.reset()

num_eps = 100
ep_cnt = 0
success = []

while (ep_cnt < num_eps):
    action = env.action_space.sample()
    _, _, done, info = env.step(action)
    if done:
        ep_cnt += 1
        success.append(info["success"])
        env.reset()

print(np.mean(success), np.std(success))
