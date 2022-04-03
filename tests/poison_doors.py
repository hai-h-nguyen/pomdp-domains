from pdomains import *
import matplotlib.pyplot as plt
import gym

env = gym.make('pdomains-poison-doors-v0')
env.reset()

for i in range(10):
    action = env.action_space.sample()
    _, _, done, _ = env.step(action)

    if done:
        print("Reset")
        env.reset()
