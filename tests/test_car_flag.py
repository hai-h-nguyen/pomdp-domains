from pdomains import *
import matplotlib.pyplot as plt
import gym
import time
import random

env=gym.make('pdomains-car-flag-symm-v0', rendering=True)
obs = env.reset()

for i in range(1000):
    action = -1
    obs, _, done, _ = env.step(action)
    time.sleep(1)
    if done:
        env.reset()
