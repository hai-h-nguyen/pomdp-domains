from pdomains import *
import matplotlib.pyplot as plt
import gym
import time

env=gym.make('pdomains-car-flag-symm-v0', rendering=True)
obs = env.reset()
time.sleep(1)

for i in range(1000):
    action = 1
    obs, reward, done, _ = env.step(action)
    print(reward)
    time.sleep(1)
    if done:
        env.reset()
        time.sleep(1)
        print()
