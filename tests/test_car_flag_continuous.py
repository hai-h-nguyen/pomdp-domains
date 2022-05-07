from pdomains import *
import matplotlib.pyplot as plt
import gym

env=gym.make('pdomains-car-flag-continuous-v0', rendering=True)
obs = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    action = env.query_expert()
    obs, _, done, _ = env.step(action)
    if done:
        env.reset()
