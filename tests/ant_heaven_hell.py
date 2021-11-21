from pdomains import *
import matplotlib.pyplot as plt
import gym

env=gym.make('pdomains-ant-heaven-hell-v0', rendering=True)
env.reset()

for i in range(1000):
    action = env.action_space.sample()
    env.step(action)
    if i % 10 == 0:
        env.reset()
