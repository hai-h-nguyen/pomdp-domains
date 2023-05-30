from pdomains import *
import matplotlib.pyplot as plt
import gym
import time
import numpy as np

env=gym.make('pdomains-peg-insertion-state-v0', rendering=True)
obs = env.reset()
time.sleep(1)

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(reward, done)
    env.render()
    # print(reward)
    if done:
        if "success" in info and info["success"] == True:
            print("Success")
        env.reset()
        time.sleep(1)
