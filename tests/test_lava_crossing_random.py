from pdomains import *
import matplotlib.pyplot as plt
import gym

env=gym.make('pdomains-lava-crossing-v0', rendering=True)
env.reset()

for i in range(1000):
    action = env.action_space.sample()
    # action = env.query_expert()
    obs, _, done, _ = env.step(env.action_space.sample())
    if done:
        env.reset()
