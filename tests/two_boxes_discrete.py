from pdomains import *
import matplotlib.pyplot as plt
import gym

env=gym.make('pdomains-two-boxes-v0', rendering=True)
env.reset()

start = 0

for i in range(1000):
    action = env.action_space.sample()
    action = env.query_expert()
    _, _, done, _= env.step(action)
    start += 1
    if done:
        env.reset()
        print("Length:", start)
        start = 0
